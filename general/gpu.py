import torch
import comfy.model_management
from ..core import logger
import os
import platform
import re
import shutil
import subprocess
import time

TEGRASTATS_PATHS = (
    'tegrastats',
    '/usr/bin/tegrastats',
    '/usr/sbin/tegrastats',
    '/bin/tegrastats',
    '/sbin/tegrastats',
)

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on a Jetson device by checking the device model
    information or the platform release.
    """
    PROC_DEVICE_MODEL = ''
    try:
        with open('/proc/device-tree/model', 'r') as f:
            PROC_DEVICE_MODEL = f.read().strip()
            logger.info(f"Device model: {PROC_DEVICE_MODEL}")
            return "NVIDIA" in PROC_DEVICE_MODEL
    except Exception as e:
        # logger.warning(f"JETSON: Could not read /proc/device-tree/model: {e} (If you're not using Jetson, ignore this warning)")
        # If /proc/device-tree/model is not available, check platform.release()
        platform_release = platform.release()
        logger.info(f"Platform release: {platform_release}")
        if 'tegra' in platform_release.lower():
            logger.info("Detected 'tegra' in platform release. Assuming Jetson device.")
            return True
        else:
            logger.info("JETSON: Not detected.")
            return False

IS_JETSON = is_jetson()

class CGPUInfo:
    """
    This class is responsible for getting information from GPU (ONLY).
    """
    cuda = False
    pynvmlLoaded = False
    jtopLoaded = False
    tegrastatsLoaded = False
    cudaAvailable = False
    torchDevice = 'cpu'
    cudaDevice = 'cpu'
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []

    def __init__(self):
        self.pynvmlLoaded = False
        self.jtopLoaded = False
        self.tegrastatsLoaded = False
        self.tegrastatsPath = None
        self.tegrastatsStats = {}
        self.lastTegrastatsUpdate = 0
        self.jtopInstance = None
        self.gpus = []
        self.gpusUtilization = []
        self.gpusVRAM = []
        self.gpusTemperature = []

        if IS_JETSON:
            # Try to import jtop for Jetson devices
            try:
                from jtop import jtop
                self.jtopInstance = jtop()
                self.jtopInstance.start()
                self.jtopLoaded = True
                logger.info('jtop initialized on Jetson device.')
            except ImportError as e:
                logger.error('jtop is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not initialize jtop. ' + str(e))

            if not self.jtopLoaded:
                self.initTegrastats()
        else:
            self.initTegrastats()

        if not (self.jtopLoaded or self.tegrastatsLoaded):
            # Try to import pynvml for non-Jetson devices
            try:
                import pynvml
                self.pynvml = pynvml
                self.pynvml.nvmlInit()
                self.pynvmlLoaded = True
                logger.info('pynvml (NVIDIA) initialized.')
            except ImportError as e:
                logger.error('pynvml is not installed. ' + str(e))
            except Exception as e:
                logger.error('Could not init pynvml (NVIDIA). ' + str(e))

        self.anygpuLoaded = self.pynvmlLoaded or self.jtopLoaded or self.tegrastatsLoaded

        try:
            self.torchDevice = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        except Exception as e:
            logger.error('Could not pick default device. ' + str(e))

        if self.pynvmlLoaded and not self.jtopLoaded and not self.deviceGetCount():
            logger.warning('No GPU detected, disabling GPU monitoring.')
            self.anygpuLoaded = False
            self.pynvmlLoaded = False
            self.jtopLoaded = False
            self.tegrastatsLoaded = False

        if self.anygpuLoaded:
            if self.deviceGetCount() > 0:
                self.cudaDevicesFound = self.deviceGetCount()

                logger.info(f"GPU/s:")

                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuName = self.deviceGetName(deviceHandle, deviceIndex)

                    logger.info(f"{deviceIndex}) {gpuName}")

                    self.gpus.append({
                        'index': deviceIndex,
                        'name': gpuName,
                    })

                    # Same index as gpus, with default values
                    self.gpusUtilization.append(True)
                    self.gpusVRAM.append(True)
                    self.gpusTemperature.append(True)

                self.cuda = True
                logger.info(self.systemGetDriverVersion())
            else:
                logger.warning('No GPU with CUDA detected.')
        else:
            logger.warning('No GPU monitoring libraries available.')

        self.cudaDevice = 'cpu' if self.torchDevice == 'cpu' else 'cuda'
        self.cudaAvailable = torch.cuda.is_available()

        if self.cuda and self.cudaAvailable and self.torchDevice == 'cpu':
            logger.warning('CUDA is available, but torch is using CPU.')

    def getInfo(self):
        logger.debug('Getting GPUs info...')
        return self.gpus

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ''
        gpus = []

        hasGpuMonitor = self.anygpuLoaded and self.cuda and self.cudaDevicesFound > 0

        if not hasGpuMonitor:
            gpuType = 'cpu'
            gpus.append({
                'gpu_utilization': -1,
                'gpu_temperature': -1,
                'vram_total': -1,
                'vram_used': -1,
                'vram_used_percent': -1,
            })
        else:
            gpuType = 'jetson' if (self.jtopLoaded or self.tegrastatsLoaded) else self.cudaDevice

            if self.tegrastatsLoaded:
                self.refreshTegrastatsStats()

            if self.anygpuLoaded and self.cuda and (self.cudaAvailable or self.jtopLoaded or self.tegrastatsLoaded):
                for deviceIndex in range(self.cudaDevicesFound):
                    deviceHandle = self.deviceGetHandleByIndex(deviceIndex)

                    gpuUtilization = -1
                    vramPercent = -1
                    vramUsed = -1
                    vramTotal = -1
                    gpuTemperature = -1

                    # GPU Utilization
                    if self.switchGPU and self.gpusUtilization[deviceIndex]:
                        try:
                            gpuUtilization = self.deviceGetUtilizationRates(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU utilization. ' + str(e))
                            logger.error('Monitor of GPU is turning off.')
                            self.switchGPU = False

                    if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                        try:
                            memory = self.deviceGetMemoryInfo(deviceHandle)
                            vramUsed = memory['used']
                            vramTotal = memory['total']

                            # Check if vramTotal is not zero or None
                            if vramTotal and vramTotal != 0:
                                vramPercent = vramUsed / vramTotal * 100
                        except Exception as e:
                            logger.error('Could not get GPU memory info. ' + str(e))
                            self.switchVRAM = False

                    # Temperature
                    if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                        try:
                            gpuTemperature = self.deviceGetTemperature(deviceHandle)
                        except Exception as e:
                            logger.error('Could not get GPU temperature. Turning off this feature. ' + str(e))
                            self.switchTemperature = False

                    gpus.append({
                        'gpu_utilization': gpuUtilization,
                        'gpu_temperature': gpuTemperature,
                        'vram_total': vramTotal,
                        'vram_used': vramUsed,
                        'vram_used_percent': vramPercent,
                    })

        return {
            'device_type': gpuType,
            'gpus': gpus,
        }

    def deviceGetCount(self):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetCount()
        elif self.jtopLoaded:
            # For Jetson devices, we assume there's one GPU
            return 1
        elif self.tegrastatsLoaded:
            return 1
        else:
            return 0

    def deviceGetHandleByIndex(self, index):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetHandleByIndex(index)
        elif self.jtopLoaded:
            return index  # On Jetson, index acts as handle
        elif self.tegrastatsLoaded:
            return index
        else:
            return 0

    def deviceGetName(self, deviceHandle, deviceIndex):
        if self.pynvmlLoaded:
            gpuName = 'Unknown GPU'

            try:
                gpuName = self.pynvml.nvmlDeviceGetName(deviceHandle)
                try:
                    gpuName = gpuName.decode('utf-8', errors='ignore')
                except AttributeError:
                    pass

            except UnicodeDecodeError as e:
                gpuName = 'Unknown GPU (decoding error)'
                logger.error(f"UnicodeDecodeError: {e}")

            return gpuName
        elif self.jtopLoaded:
            # Access the GPU name from self.jtopInstance.gpu
            try:
                gpu_info = self.jtopInstance.gpu
                gpu_name = next(iter(gpu_info.keys()))
                return gpu_name
            except Exception as e:
                logger.error('Could not get GPU name. ' + str(e))
                return 'Unknown GPU'
        elif self.tegrastatsLoaded:
            return self.getJetsonGpuName()
        else:
            return ''

    def systemGetDriverVersion(self):
        if self.pynvmlLoaded:
            return f'NVIDIA Driver: {self.pynvml.nvmlSystemGetDriverVersion()}'
        elif self.jtopLoaded:
            # No direct method to get driver version from jtop
            return 'NVIDIA Driver: unknown'
        elif self.tegrastatsLoaded:
            return 'NVIDIA Jetson tegrastats initialized'
        else:
            return 'Driver unknown'

    def deviceGetUtilizationRates(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetUtilizationRates(deviceHandle).gpu
        elif self.jtopLoaded:
            # GPU utilization from jtop stats
            try:
                gpu_util = self.jtopInstance.stats.get('GPU', -1)
                return gpu_util
            except Exception as e:
                logger.error('Could not get GPU utilization. ' + str(e))
                return -1
        elif self.tegrastatsLoaded:
            return self.tegrastatsStats.get('gpu_utilization', -1)
        else:
            return 0

    def deviceGetMemoryInfo(self, deviceHandle):
        if self.pynvmlLoaded:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
            return {'total': mem.total, 'used': mem.used}
        elif self.jtopLoaded:
            mem_data = self.jtopInstance.memory['RAM']
            total = mem_data['tot']
            used = mem_data['used']
            return {'total': total, 'used': used}
        elif self.tegrastatsLoaded:
            return {
                'total': self.tegrastatsStats.get('ram_total', -1),
                'used': self.tegrastatsStats.get('ram_used', -1),
            }
        else:
            return {'total': 1, 'used': 1}

    def deviceGetTemperature(self, deviceHandle):
        if self.pynvmlLoaded:
            return self.pynvml.nvmlDeviceGetTemperature(deviceHandle, self.pynvml.NVML_TEMPERATURE_GPU)
        elif self.jtopLoaded:
            try:
                temperature = self.jtopInstance.stats.get('Temp gpu', -1)
                return temperature
            except Exception as e:
                logger.error('Could not get GPU temperature. ' + str(e))
                return -1
        elif self.tegrastatsLoaded:
            return self.tegrastatsStats.get('gpu_temperature', -1)
        else:
            return 0

    def close(self):
        if self.jtopLoaded and self.jtopInstance is not None:
            self.jtopInstance.close()

    def initTegrastats(self):
        self.tegrastatsPath = self.findTegrastats()
        if not self.tegrastatsPath:
            logger.error('tegrastats is not installed or not available in PATH.')
            return

        line = self.readTegrastatsLine()
        if not line:
            logger.error('Could not read from tegrastats.')
            return

        self.tegrastatsLoaded = True
        self.tegrastatsStats = self.parseTegrastatsLine(line)
        self.lastTegrastatsUpdate = time.monotonic()
        logger.info('tegrastats initialized on Jetson device.')

    def findTegrastats(self):
        env_path = os.environ.get('CRYSTOOLS_TEGRASTATS_PATH')
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path

        for path in TEGRASTATS_PATHS:
            resolved = shutil.which(path) if '/' not in path else path
            if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
                return resolved

        return None

    def refreshTegrastatsStats(self):
        if not self.tegrastatsLoaded:
            return

        line = self.readTegrastatsLine()
        if not line:
            return

        self.tegrastatsStats = self.parseTegrastatsLine(line)
        self.lastTegrastatsUpdate = time.monotonic()

    def readTegrastatsLine(self):
        if not self.tegrastatsPath:
            return ''

        process = None
        try:
            # Start tegrastats (continuous mode)
            process = subprocess.Popen(
                [self.tegrastatsPath, '--interval', '1000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            # Read only the first valid line
            start = time.time()
            while True:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        return line

                # Safety timeout (2s)
                if time.time() - start > 2:
                    break

        except Exception as e:
            logger.error('Could not read tegrastats. ' + str(e))

        finally:
            if process:
                try:
                    process.kill()
                except Exception:
                    pass

        return ''

    def parseTegrastatsLine(self, line):
        stats = {
            'gpu_utilization': -1,
            'gpu_temperature': -1,
            'cpu_temperature': -1,
            'ram_total': -1,
            'ram_used': -1,
        }

        tokens = line.split()

        for i, tok in enumerate(tokens):
            # GPU usage
            if tok == "GR3D_FREQ" and i + 1 < len(tokens):
                val = tokens[i + 1].split('%')[0]
                if val.replace('.', '', 1).isdigit():
                    stats['gpu_utilization'] = float(val)

            # GPU temp (gpu@53.5C or GPU@53.5C)
            if tok.lower().startswith("gpu@"):
                try:
                    stats['gpu_temperature'] = float(tok.split('@')[1].replace('C', ''))
                except:
                    pass

            # CPU temp
            if tok.lower().startswith("cpu@"):
                try:
                    stats['cpu_temperature'] = float(tok.split('@')[1].replace('C', ''))
                except:
                    pass

            # RAM
            if tok == "RAM" and i + 1 < len(tokens):
                try:
                    used, total = tokens[i + 1].split('/')
                    unit = tokens[i + 1][-2:]  # MB / GB
                    multiplier = self.memoryUnitMultiplier(unit)

                    stats['ram_used'] = int(float(used) * multiplier)
                    stats['ram_total'] = int(float(total[:-2]) * multiplier)
                except:
                    pass

        return stats

    def memoryUnitMultiplier(self, unit):
        multipliers = {
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
        }
        return multipliers.get(unit, 1)

    def getJetsonGpuName(self):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().replace('\x00', '')
                return f'{model} GPU'
        except Exception:
            return 'NVIDIA Jetson GPU'
