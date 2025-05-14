"""
시뮬레이션 데이터 모델 모듈

이 모듈은 표준 용액 기반 비침습적 체외 진단 시뮬레이션에 사용되는
데이터 모델 클래스들을 정의합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime


@dataclass
class SensorConfig:
    """센서 구성 정보"""
    sensor_id: str
    sampling_rate: float = 1.0
    noise_level: float = 0.02
    sensitivity: float = 0.85
    drift_rate: float = 0.001
    response_time: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            'sensor_id': self.sensor_id,
            'sampling_rate': self.sampling_rate,
            'noise_level': self.noise_level,
            'sensitivity': self.sensitivity,
            'drift_rate': self.drift_rate,
            'response_time': self.response_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorConfig':
        """딕셔너리에서 객체 생성"""
        return cls(
            sensor_id=data['sensor_id'],
            sampling_rate=data.get('sampling_rate', 1.0),
            noise_level=data.get('noise_level', 0.02),
            sensitivity=data.get('sensitivity', 0.85),
            drift_rate=data.get('drift_rate', 0.001),
            response_time=data.get('response_time', 3.0)
        )


@dataclass
class SensorReading:
    """센서 측정값"""
    timestamp: float
    value: float
    sensor_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'sensor_id': self.sensor_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """딕셔너리에서 객체 생성"""
        return cls(
            timestamp=data['timestamp'],
            value=data['value'],
            sensor_id=data['sensor_id']
        )


@dataclass
class SkinProperties:
    """피부 특성 데이터"""
    thickness: float = 2.0  # mm
    permeability: float = 0.5  # 0~1
    sweat_rate: float = 1.0  # 0~10
    
    def to_dict(self) -> Dict[str, float]:
        """객체를 딕셔너리로 변환"""
        return {
            'thickness': self.thickness,
            'permeability': self.permeability,
            'sweat_rate': self.sweat_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SkinProperties':
        """딕셔너리에서 객체 생성"""
        return cls(
            thickness=data.get('thickness', 2.0),
            permeability=data.get('permeability', 0.5),
            sweat_rate=data.get('sweat_rate', 1.0)
        )


@dataclass
class EnvironmentConditions:
    """환경 조건 데이터"""
    temperature: float = 25.0  # °C
    humidity: float = 50.0  # %
    ph: float = 6.5  # pH
    
    def to_dict(self) -> Dict[str, float]:
        """객체를 딕셔너리로 변환"""
        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'ph': self.ph
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EnvironmentConditions':
        """딕셔너리에서 객체 생성"""
        return cls(
            temperature=data.get('temperature', 25.0),
            humidity=data.get('humidity', 50.0),
            ph=data.get('ph', 6.5)
        )


@dataclass
class SimulationParameters:
    """시뮬레이션 파라미터"""
    blood_glucose: float = 120.0  # mg/dL
    standard_concentration: float = 100.0  # mg/dL
    salt_resistance: float = 500.0  # Ω
    simulation_duration: int = 300  # 초
    sampling_interval: int = 10  # 초
    skin_properties: SkinProperties = field(default_factory=SkinProperties)
    environment: EnvironmentConditions = field(default_factory=EnvironmentConditions)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후처리"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            'blood_glucose': self.blood_glucose,
            'standard_concentration': self.standard_concentration,
            'salt_resistance': self.salt_resistance,
            'simulation_duration': self.simulation_duration,
            'sampling_interval': self.sampling_interval,
            'skin_properties': self.skin_properties.to_dict(),
            'environment': self.environment.to_dict(),
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        """딕셔너리에서 객체 생성"""
        skin_props = SkinProperties.from_dict(data.get('skin_properties', {}))
        env_conds = EnvironmentConditions.from_dict(data.get('environment', {}))
        
        return cls(
            blood_glucose=data.get('blood_glucose', 120.0),
            standard_concentration=data.get('standard_concentration', 100.0),
            salt_resistance=data.get('salt_resistance', 500.0),
            simulation_duration=data.get('simulation_duration', 300),
            sampling_interval=data.get('sampling_interval', 10),
            skin_properties=skin_props,
            environment=env_conds,
            timestamp=data.get('timestamp')
        )


@dataclass
class CalibrationResult:
    """보정 결과 데이터"""
    uncalibrated_value: float
    calibrated_value: float
    calibration_factor: float
    diffusion_rate: float
    r_squared: float = 0.0
    reference_value: Optional[float] = None
    error_uncalibrated: Optional[float] = None
    error_calibrated: Optional[float] = None
    improvement: Optional[float] = None
    unit: str = 'mg/dL'
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            'uncalibrated_value': self.uncalibrated_value,
            'calibrated_value': self.calibrated_value,
            'calibration_factor': self.calibration_factor,
            'diffusion_rate': self.diffusion_rate,
            'r_squared': self.r_squared,
            'reference_value': self.reference_value,
            'error_uncalibrated': self.error_uncalibrated,
            'error_calibrated': self.error_calibrated,
            'improvement': self.improvement,
            'unit': self.unit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """딕셔너리에서 객체 생성"""
        return cls(
            uncalibrated_value=data['uncalibrated_value'],
            calibrated_value=data['calibrated_value'],
            calibration_factor=data['calibration_factor'],
            diffusion_rate=data['diffusion_rate'],
            r_squared=data.get('r_squared', 0.0),
            reference_value=data.get('reference_value'),
            error_uncalibrated=data.get('error_uncalibrated'),
            error_calibrated=data.get('error_calibrated'),
            improvement=data.get('improvement'),
            unit=data.get('unit', 'mg/dL')
        )


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    parameters: SimulationParameters
    sensor_data: Dict[str, List[Dict[str, Any]]]
    analysis_results: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후처리"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            'parameters': self.parameters.to_dict(),
            'sensor_data': self.sensor_data,
            'analysis_results': self.analysis_results,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResult':
        """딕셔너리에서 객체 생성"""
        params = SimulationParameters.from_dict(data.get('parameters', {}))
        
        return cls(
            parameters=params,
            sensor_data=data.get('sensor_data', {}),
            analysis_results=data.get('analysis_results', {}),
            timestamp=data.get('timestamp')
        )
