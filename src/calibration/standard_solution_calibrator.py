"""
표준 용액 기반 보정 알고리즘 구현

이 모듈은 비침습적 체외 진단을 위한 표준 용액 기반 보정 알고리즘을 구현합니다.
미지의 포도당 농도를 측정하는 제1센서와 알려진 농도의 표준 용액을 사용하는 제2센서의 
데이터를 기반으로 보정 계수를 계산하고 최종 포도당 농도를 도출합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

class StandardSolutionCalibrator:
    """표준 용액 기반 보정 알고리즘을 구현하는 클래스"""
    
    def __init__(self, 
                 standard_concentration: float, 
                 sensor_type: str = 'glucose',
                 temp_correction: bool = True,
                 ph_correction: bool = True):
        """
        초기화 함수
        
        파라미터:
            standard_concentration (float): 표준 용액의 초기 농도 (mg/dL)
            sensor_type (str): 센서 타입 ('glucose' 또는 'salt')
            temp_correction (bool): 온도 보정 활성화 여부
            ph_correction (bool): pH 보정 활성화 여부
        """
        self.standard_concentration = standard_concentration
        self.sensor_type = sensor_type
        self.temp_correction = temp_correction
        self.ph_correction = ph_correction
        self.time_points = []
        self.standard_values = []
        self.diffusion_rate = None
        self.calibration_factor = 1.0
        
    def add_standard_measurement(self, time_point: int, measured_value: float) -> None:
        """
        표준 용액 측정값 추가
        
        파라미터:
            time_point (int): 측정 시점 (초)
            measured_value (float): 측정된 표준 용액 값
        """
        self.time_points.append(time_point)
        self.standard_values.append(measured_value)
        
    def calculate_diffusion_rate(self) -> float:
        """
        확산율 계산
        
        반환값:
            float: 계산된 확산율
        """
        if len(self.time_points) < 2:
            raise ValueError("최소 2개 이상의 측정점이 필요합니다")
            
        # 시간에 따른 농도 변화 기울기 계산
        x = np.array(self.time_points)
        y = np.array(self.standard_values)
        
        slope, _, _, _, _ = stats.linregress(x, y)
        self.diffusion_rate = slope
        
        return self.diffusion_rate
    
    def compute_calibration_factor(self, 
                                  skin_thickness: Optional[float] = None,
                                  temperature: Optional[float] = None,
                                  ph_value: Optional[float] = None) -> float:
        """
        보정 계수 계산
        
        파라미터:
            skin_thickness (float, optional): 피부 두께 (mm)
            temperature (float, optional): 측정 환경 온도 (섭씨)
            ph_value (float, optional): 땀의 pH 값
            
        반환값:
            float: 계산된 보정 계수
        """
        if self.diffusion_rate is None:
            self.calculate_diffusion_rate()
            
        # 확산율 기반 기본 보정 계수 계산
        initial_value = self.standard_values[0]
        final_value = self.standard_values[-1]
        total_change = abs(final_value - initial_value)
        change_ratio = total_change / self.standard_concentration
        
        # 기본 보정 계수
        self.calibration_factor = 1.0 + change_ratio
        
        # 피부 두께 보정
        if skin_thickness is not None:
            thickness_factor = 1.0 + (skin_thickness - 2.0) * 0.05  # 기준 두께 2.0mm
            self.calibration_factor *= thickness_factor
            
        # 온도 보정
        if self.temp_correction and temperature is not None:
            temp_factor = 1.0 + (temperature - 25.0) * 0.01  # 기준 온도 25도
            self.calibration_factor *= temp_factor
            
        # pH 보정
        if self.ph_correction and ph_value is not None:
            ph_factor = 1.0 + (ph_value - 7.0) * 0.03  # 기준 pH 7.0
            self.calibration_factor *= ph_factor
            
        return self.calibration_factor
    
    def calibrate_measurement(self, measured_value: float) -> float:
        """
        실제 측정값 보정
        
        파라미터:
            measured_value (float): 보정할 측정값
            
        반환값:
            float: 보정된 측정값
        """
        if self.calibration_factor is None:
            raise ValueError("보정 계수 계산이 필요합니다. compute_calibration_factor()를 먼저 호출하세요.")
            
        return measured_value * self.calibration_factor
    
    def plot_standard_curve(self) -> plt.Figure:
        """
        표준 곡선 시각화
        
        반환값:
            matplotlib.figure.Figure: 생성된 그래프
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.time_points, self.standard_values, 'o-', label='표준 용액 농도')
        ax.set_xlabel('시간 (초)')
        
        if self.sensor_type == 'glucose':
            ax.set_ylabel('농도 (mg/dL)')
            ax.set_title('시간에 따른 표준 포도당 용액 농도 변화')
        else:  # salt
            ax.set_ylabel('저항 (Ω)')
            ax.set_title('시간에 따른 표준 소금물 저항 변화')
            
        # 확산율 추세선
        if self.diffusion_rate is not None:
            x = np.array(self.time_points)
            y_pred = self.diffusion_rate * x + self.standard_values[0]
            ax.plot(x, y_pred, '--', color='red', label=f'확산율: {self.diffusion_rate:.4f} 단위/초')
            
        ax.grid(True)
        ax.legend()
        
        return fig


class DualSensorCalibrator:
    """이중 센서 시스템에서 포도당 농도 측정 및 보정을 처리하는 클래스"""
    
    def __init__(self, standard_glucose_concentration: float = 100.0):
        """
        초기화 함수
        
        파라미터:
            standard_glucose_concentration (float): 표준 포도당 용액의 초기 농도 (mg/dL)
        """
        self.glucose_calibrator = StandardSolutionCalibrator(
            standard_concentration=standard_glucose_concentration,
            sensor_type='glucose'
        )
        self.salt_calibrator = None
        self.primary_calibrator = self.glucose_calibrator
        
    def use_salt_solution(self, standard_resistance: float) -> None:
        """
        소금물을 표준 용액으로 사용
        
        파라미터:
            standard_resistance (float): 표준 소금물의 초기 저항값 (Ω)
        """
        self.salt_calibrator = StandardSolutionCalibrator(
            standard_concentration=standard_resistance,
            sensor_type='salt'
        )
        # 기본 보정기를 소금물로 설정
        self.primary_calibrator = self.salt_calibrator
        
    def add_standard_measurements(self, 
                                 time_points: List[int], 
                                 standard_values: List[float], 
                                 solution_type: str = 'primary') -> None:
        """
        표준 용액 측정값 일괄 추가
        
        파라미터:
            time_points (List[int]): 측정 시점 리스트 (초)
            standard_values (List[float]): 표준 용액 측정값 리스트
            solution_type (str): 측정 용액 타입 ('primary', 'glucose', 'salt')
        """
        if solution_type == 'primary':
            calibrator = self.primary_calibrator
        elif solution_type == 'glucose':
            calibrator = self.glucose_calibrator
        elif solution_type == 'salt':
            if self.salt_calibrator is None:
                raise ValueError("소금물 보정기가 초기화되지 않았습니다. use_salt_solution()을 먼저 호출하세요.")
            calibrator = self.salt_calibrator
        else:
            raise ValueError("지원되지 않는 용액 타입입니다. 'primary', 'glucose', 또는 'salt'를 사용하세요.")
            
        for time_point, value in zip(time_points, standard_values):
            calibrator.add_standard_measurement(time_point, value)
            
    def calculate_calibration_factor(self, 
                                   skin_thickness: Optional[float] = None,
                                   temperature: Optional[float] = None,
                                   ph_value: Optional[float] = None) -> Dict[str, float]:
        """
        모든 활성화된 보정기에 대해 보정 계수 계산
        
        파라미터:
            skin_thickness (float, optional): 피부 두께 (mm)
            temperature (float, optional): 측정 환경 온도 (섭씨)
            ph_value (float, optional): 땀의 pH 값
            
        반환값:
            Dict[str, float]: 각 보정기별 보정 계수
        """
        result = {}
        
        # 포도당 보정기
        result['glucose'] = self.glucose_calibrator.compute_calibration_factor(
            skin_thickness=skin_thickness,
            temperature=temperature,
            ph_value=ph_value
        )
        
        # 소금물 보정기 (사용 중인 경우)
        if self.salt_calibrator is not None:
            result['salt'] = self.salt_calibrator.compute_calibration_factor(
                skin_thickness=skin_thickness,
                temperature=temperature,
                ph_value=ph_value
            )
            
        return result
    
    def calibrate_glucose_measurement(self, measured_value: float) -> float:
        """
        포도당 측정값 보정
        
        파라미터:
            measured_value (float): 보정할 포도당 측정값
            
        반환값:
            float: 보정된 포도당 측정값
        """
        return self.primary_calibrator.calibrate_measurement(measured_value)
    
    def analyze_data(self, uncalibrated_value: float) -> Dict[str, Union[float, str]]:
        """
        측정 데이터 분석 및 보정
        
        파라미터:
            uncalibrated_value (float): 미보정 포도당 측정값
            
        반환값:
            Dict[str, Union[float, str]]: 분석 결과
        """
        # 보정 계수가 계산되지 않은 경우
        if self.primary_calibrator.calibration_factor is None:
            self.calculate_calibration_factor()
            
        # 확산율 계산
        diffusion_rate = self.primary_calibrator.diffusion_rate
        
        # 보정 계수
        calibration_factor = self.primary_calibrator.calibration_factor
        
        # 보정된 포도당 값
        calibrated_value = self.calibrate_glucose_measurement(uncalibrated_value)
        
        # 분석 결과
        result = {
            'uncalibrated_value': uncalibrated_value,
            'calibrated_value': calibrated_value,
            'calibration_factor': calibration_factor,
            'diffusion_rate': diffusion_rate,
            'confidence_level': self._calculate_confidence_level(diffusion_rate)
        }
        
        return result
    
    def _calculate_confidence_level(self, diffusion_rate: float) -> str:
        """
        확산율 기반 신뢰도 수준 계산 (내부 함수)
        
        파라미터:
            diffusion_rate (float): 계산된 확산율
            
        반환값:
            str: 신뢰도 수준 ('높음', '중간', '낮음')
        """
        abs_rate = abs(diffusion_rate)
        
        if abs_rate < 0.05:
            return '높음'
        elif abs_rate < 0.1:
            return '중간'
        else:
            return '낮음'


# 코드 사용 예시
if __name__ == "__main__":
    # 예제 데이터
    time_points = [0, 60, 120, 180, 240, 300]
    
    # 포도당 용액 데이터 (표 1, 샘플 1)
    glucose_values = [100, 98, 96, 94, 92, 90]
    
    # 소금물 저항 데이터 (표 2, 샘플 1)
    salt_resistance = [500, 482, 465, 441, 420, 403]
    
    # 이중 센서 보정기 초기화
    dual_calibrator = DualSensorCalibrator(standard_glucose_concentration=100.0)
    
    # 소금물 보정기 추가
    dual_calibrator.use_salt_solution(standard_resistance=500.0)
    
    # 표준 용액 측정값 추가
    dual_calibrator.add_standard_measurements(time_points, glucose_values, solution_type='glucose')
    dual_calibrator.add_standard_measurements(time_points, salt_resistance, solution_type='salt')
    
    # 보정 계수 계산
    calibration_factors = dual_calibrator.calculate_calibration_factor(
        temperature=25.0,
        ph_value=6.5
    )
    
    print(f"보정 계수: {calibration_factors}")
    
    # 측정 데이터 분석
    result = dual_calibrator.analyze_data(uncalibrated_value=125)
    
    print("\n측정 결과 분석:")
    print(f"미보정 값: {result['uncalibrated_value']} mg/dL")
    print(f"보정된 값: {result['calibrated_value']:.1f} mg/dL")
    print(f"보정 계수: {result['calibration_factor']:.4f}")
    print(f"확산율: {result['diffusion_rate']:.6f}")
    print(f"신뢰도: {result['confidence_level']}")
