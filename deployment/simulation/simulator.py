"""
표준 용액 기반 비침습적 체외 진단 시뮬레이터

이 모듈은 표준 용액을 이용한 비침습적 체외 진단 방법을 시뮬레이션하는 도구를 제공합니다.
센서 시뮬레이션, 표준 용액의 농도 변화, 보정 알고리즘 등의 기능을 포함합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import time
from datetime import datetime
import json
import os
import sys

# 상위 디렉토리의 모듈 가져오기 위한 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 프로젝트 모듈 임포트
from src.calibration.standard_solution_calibrator import StandardSolutionCalibrator, DualSensorCalibrator
from src.sensors.glucose_sensor import GlucoseSensor, SaltSolutionSensor, DualSensorSystem
from src.analysis.glucose_data_analyzer import GlucoseDataAnalyzer


class GlucoseMonitoringSimulator:
    """표준 용액 기반 비침습적 체외 진단 시뮬레이터 클래스"""
    
    def __init__(self, 
                blood_glucose: float = 120.0,
                standard_concentration: float = 100.0,
                salt_resistance: float = 500.0,
                simulation_duration: int = 300,
                sampling_interval: int = 10):
        """
        초기화 함수
        
        파라미터:
            blood_glucose (float): 시뮬레이션할 실제 혈당값 (mg/dL)
            standard_concentration (float): 표준 포도당 용액 농도 (mg/dL)
            salt_resistance (float): 표준 소금물 초기 저항 (Ω)
            simulation_duration (int): 시뮬레이션 기간 (초)
            sampling_interval (int): 샘플링 간격 (초)
        """
        self.blood_glucose = blood_glucose
        self.standard_concentration = standard_concentration
        self.salt_resistance = salt_resistance
        self.simulation_duration = simulation_duration
        self.sampling_interval = sampling_interval
        
        # 센서 특성 설정
        self.skin_properties = {
            'thickness': 2.0,  # mm
            'permeability': 0.5,  # 0~1
            'sweat_rate': 1.0,  # 0~10
        }
        
        # 환경 조건
        self.environment = {
            'temperature': 25.0,  # °C
            'humidity': 50.0,  # %
            'ph': 6.5  # pH
        }
        
        # 센서 시스템
        self.sensor_system = None
        
        # 시뮬레이션 결과
        self.simulation_results = {
            'sensor_data': None,
            'analysis_results': None,
            'parameters': {
                'blood_glucose': blood_glucose,
                'standard_concentration': standard_concentration,
                'salt_resistance': salt_resistance,
                'simulation_duration': simulation_duration,
                'sampling_interval': sampling_interval,
                'skin_properties': dict(self.skin_properties),
                'environment': dict(self.environment)
            }
        }
        
    def set_skin_properties(self, thickness: float, permeability: float, sweat_rate: float) -> None:
        """
        피부 특성 설정
        
        파라미터:
            thickness (float): 피부 두께 (mm)
            permeability (float): 투과성 (0~1)
            sweat_rate (float): 땀 분비 속도 (0~10)
        """
        self.skin_properties['thickness'] = thickness
        self.skin_properties['permeability'] = permeability
        self.skin_properties['sweat_rate'] = sweat_rate
        
        # 파라미터 업데이트
        self.simulation_results['parameters']['skin_properties'] = dict(self.skin_properties)
        
    def set_environment(self, temperature: float, humidity: float, ph: float) -> None:
        """
        환경 조건 설정
        
        파라미터:
            temperature (float): 온도 (°C)
            humidity (float): 습도 (%)
            ph (float): pH
        """
        self.environment['temperature'] = temperature
        self.environment['humidity'] = humidity
        self.environment['ph'] = ph
        
        # 파라미터 업데이트
        self.simulation_results['parameters']['environment'] = dict(self.environment)
        
    def run_simulation(self, real_time: bool = False) -> Dict[str, Any]:
        """
        시뮬레이션 실행
        
        파라미터:
            real_time (bool): 실시간 시뮬레이션 여부
            
        반환값:
            Dict[str, Any]: 시뮬레이션 결과
        """
        print(f"표준 용액 기반 비침습적 체외 진단 시뮬레이션 시작...")
        print(f"혈당값: {self.blood_glucose} mg/dL")
        print(f"표준 포도당 용액: {self.standard_concentration} mg/dL")
        print(f"표준 소금물 저항: {self.salt_resistance} Ω")
        print(f"시뮬레이션 기간: {self.simulation_duration}초")
        print(f"샘플링 간격: {self.sampling_interval}초")
        print()
        
        # 센서 시스템 초기화
        self.sensor_system = DualSensorSystem(
            standard_concentration=self.standard_concentration,
            initial_resistance=self.salt_resistance
        )
        
        # 실제 혈당에 따른 센서 기본값 조정
        # 실제 센서가 측정하는 값은 혈당보다 낮은 경향이 있음 (약 5~15% 낮게 측정)
        sweat_glucose_ratio = 0.85 + 0.1 * self.skin_properties['permeability']
        primary_baseline = self.blood_glucose * sweat_glucose_ratio
        
        self.sensor_system.primary_sensor.baseline_value = primary_baseline
        
        # 피부 특성에 따른 센서 감도 조정
        permeability_factor = self.skin_properties['permeability']
        self.sensor_system.primary_sensor.sensitivity = 0.7 + 0.3 * permeability_factor
        self.sensor_system.standard_glucose_sensor.sensitivity = 0.7 + 0.3 * permeability_factor
        self.sensor_system.salt_sensor.sensitivity = 0.7 + 0.3 * permeability_factor
        
        # 땀 분비 속도에 따른 센서 드리프트 조정
        sweat_factor = self.skin_properties['sweat_rate'] / 5.0
        self.sensor_system.primary_sensor.drift_rate = 0.001 * (1.0 + sweat_factor)
        self.sensor_system.standard_glucose_sensor.drift_rate = 0.001 * (1.0 + sweat_factor)
        self.sensor_system.salt_sensor.drift_rate = 0.0005 * (1.0 + sweat_factor)
        
        # 피부 두께에 따른 센서 응답 시간 조정
        thickness_factor = self.skin_properties['thickness'] / 2.0
        self.sensor_system.primary_sensor.response_time = 3.0 * thickness_factor
        self.sensor_system.standard_glucose_sensor.response_time = 3.0 * thickness_factor
        self.sensor_system.salt_sensor.response_time = 2.0 * thickness_factor
        
        # 노이즈 레벨 설정 (환경 조건에 따라)
        temp_factor = abs(self.environment['temperature'] - 25.0) / 10.0
        humidity_factor = abs(self.environment['humidity'] - 50.0) / 30.0
        noise_level = 0.02 + 0.03 * (temp_factor + humidity_factor)
        
        self.sensor_system.primary_sensor.noise_level = noise_level
        self.sensor_system.standard_glucose_sensor.noise_level = noise_level
        self.sensor_system.salt_sensor.noise_level = noise_level * 0.5
        
        # 센서 시작
        self.sensor_system.start_all_sensors()
        
        # 측정 데이터 수집
        time_points = []
        primary_values = []
        std_glucose_values = []
        std_salt_values = []
        
        start_time = time.time()
        elapsed_time = 0
        
        try:
            while elapsed_time < self.simulation_duration:
                # 현재 센서값 가져오기
                latest_values = self.sensor_system.get_latest_values()
                
                if all(v is not None for v in latest_values.values()):
                    time_points.append(elapsed_time)
                    primary_values.append(latest_values['primary'])
                    std_glucose_values.append(latest_values['std_glucose'])
                    std_salt_values.append(latest_values['std_salt'])
                    
                    # 실시간 모드에서는 현재 값 출력
                    if real_time and elapsed_time % 30 == 0:
                        print(f"[{elapsed_time}초] 측정값: "
                              f"기본 센서={latest_values['primary']:.1f} mg/dL, "
                              f"표준 포도당={latest_values['std_glucose']:.1f} mg/dL, "
                              f"표준 소금물={latest_values['std_salt']:.1f} Ω")
                
                # 실제 시간 모드인 경우 실제 시간만큼 대기
                if real_time:
                    time.sleep(self.sampling_interval)
                    elapsed_time = time.time() - start_time
                else:
                    # 빠른 시뮬레이션 모드
                    elapsed_time += self.sampling_interval
                    time.sleep(0.01)  # CPU 부하 방지
                
        finally:
            # 센서 중지
            self.sensor_system.stop_all_sensors()
        
        # 시뮬레이션 완료 메시지
        print("\n시뮬레이션 완료!")
        
        # 분석을 위한 데이터 형식으로 변환
        sensor_data = {
            "primary": [
                {"timestamp": start_time + t, "value": v, "sensor_id": "primary"}
                for t, v in zip(time_points, primary_values)
            ],
            "std_glucose": [
                {"timestamp": start_time + t, "value": v, "sensor_id": "std_glucose"}
                for t, v in zip(time_points, std_glucose_values)
            ],
            "std_salt": [
                {"timestamp": start_time + t, "value": v, "sensor_id": "std_salt"}
                for t, v in zip(time_points, std_salt_values)
            ]
        }
        
        # 결과 저장
        self.simulation_results['sensor_data'] = sensor_data
        
        # 보정 알고리즘 분석
        self._analyze_calibration()
        
        return self.simulation_results
    
    def _analyze_calibration(self) -> None:
        """시뮬레이션 결과 분석 (내부 메서드)"""
        analyzer = GlucoseDataAnalyzer()
        analyzer.measurement_data = self.simulation_results['sensor_data']
        
        # 보정 전후 비교
        result_glucose = analyzer.compare_calibrated_vs_uncalibrated(
            primary_sensor_id="primary",
            standard_sensor_id="std_glucose",
            reference_value=self.blood_glucose
        )
        
        result_salt = analyzer.compare_calibrated_vs_uncalibrated(
            primary_sensor_id="primary",
            standard_sensor_id="std_salt",
            reference_value=self.blood_glucose
        )
        
        # 분석 결과 저장
        self.simulation_results['analysis_results'] = {
            'glucose_calibration': result_glucose,
            'salt_calibration': result_salt
        }
        
        # 결과 출력
        print("\n분석 결과:")
        print("\n[표준 포도당 용액 기반 보정]")
        print(f"보정 전 측정값: {result_glucose['uncalibrated_value']:.1f} mg/dL")
        print(f"보정 후 측정값: {result_glucose['calibrated_value']:.1f} mg/dL")
        print(f"실제 혈당값: {self.blood_glucose:.1f} mg/dL")
        print(f"보정 전 오차: {result_glucose['error_uncalibrated']:.2f}%")
        print(f"보정 후 오차: {result_glucose['error_calibrated']:.2f}%")
        print(f"개선율: {result_glucose['improvement']:.2f}%")
        
        print("\n[표준 소금물 기반 보정]")
        print(f"보정 전 측정값: {result_salt['uncalibrated_value']:.1f} mg/dL")
        print(f"보정 후 측정값: {result_salt['calibrated_value']:.1f} mg/dL")
        print(f"실제 혈당값: {self.blood_glucose:.1f} mg/dL")
        print(f"보정 전 오차: {result_salt['error_uncalibrated']:.2f}%")
        print(f"보정 후 오차: {result_salt['error_calibrated']:.2f}%")
        print(f"개선율: {result_salt['improvement']:.2f}%")
    
    def generate_report(self, output_dir: str = '.') -> Dict[str, str]:
        """
        시뮬레이션 결과 보고서 생성
        
        파라미터:
            output_dir (str): 출력 디렉토리 경로
            
        반환값:
            Dict[str, str]: 생성된 파일 경로
        """
        if self.simulation_results['sensor_data'] is None:
            raise ValueError("시뮬레이션을 먼저 실행해야 합니다.")
            
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 파일 경로
        json_path = os.path.join(output_dir, f"simulation_results_{timestamp}.json")
        glucose_plot_path = os.path.join(output_dir, f"glucose_sensors_{timestamp}.png")
        salt_plot_path = os.path.join(output_dir, f"salt_sensor_{timestamp}.png")
        comparison_plot_path = os.path.join(output_dir, f"calibration_comparison_{timestamp}.png")
        
        # JSON 결과 저장
        with open(json_path, 'w') as f:
            json.dump(self.simulation_results, f, indent=2)
            
        # 분석기 초기화
        analyzer = GlucoseDataAnalyzer()
        analyzer.measurement_data = self.simulation_results['sensor_data']
        
        # 시계열 그래프
        fig1 = analyzer.plot_time_series(
            sensor_ids=["primary", "std_glucose"],
            title="포도당 센서 측정값 시계열"
        )
        fig1.savefig(glucose_plot_path)
        
        fig2 = analyzer.plot_time_series(
            sensor_ids=["std_salt"],
            title="소금물 센서 측정값 시계열"
        )
        fig2.savefig(salt_plot_path)
        
        # 보정 비교 그래프
        # 두 교정 방법 중 더 나은 결과를 사용
        glucose_improvement = self.simulation_results['analysis_results']['glucose_calibration']['improvement']
        salt_improvement = self.simulation_results['analysis_results']['salt_calibration']['improvement']
        
        if glucose_improvement > salt_improvement:
            better_result = self.simulation_results['analysis_results']['glucose_calibration']
            calibration_method = "포도당 표준 용액"
        else:
            better_result = self.simulation_results['analysis_results']['salt_calibration']
            calibration_method = "소금물 표준 용액"
            
        fig3 = analyzer.plot_calibration_comparison(better_result)
        fig3.suptitle(f"보정 전후 측정값 비교 ({calibration_method} 사용)")
        fig3.savefig(comparison_plot_path)
        
        plt.close('all')
        
        # 결과 경로 반환
        return {
            'json': json_path,
            'glucose_plot': glucose_plot_path,
            'salt_plot': salt_plot_path,
            'comparison_plot': comparison_plot_path
        }
    
    def save_simulation_parameters(self, filepath: str) -> None:
        """
        시뮬레이션 파라미터 저장
        
        파라미터:
            filepath (str): 저장할 파일 경로
        """
        params = {
            'blood_glucose': self.blood_glucose,
            'standard_concentration': self.standard_concentration,
            'salt_resistance': self.salt_resistance,
            'simulation_duration': self.simulation_duration,
            'sampling_interval': self.sampling_interval,
            'skin_properties': self.skin_properties,
            'environment': self.environment,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
            
    def load_simulation_parameters(self, filepath: str) -> None:
        """
        시뮬레이션 파라미터 로드
        
        파라미터:
            filepath (str): 로드할 파일 경로
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
            
        self.blood_glucose = params.get('blood_glucose', self.blood_glucose)
        self.standard_concentration = params.get('standard_concentration', self.standard_concentration)
        self.salt_resistance = params.get('salt_resistance', self.salt_resistance)
        self.simulation_duration = params.get('simulation_duration', self.simulation_duration)
        self.sampling_interval = params.get('sampling_interval', self.sampling_interval)
        
        if 'skin_properties' in params:
            self.skin_properties = params['skin_properties']
            
        if 'environment' in params:
            self.environment = params['environment']
            
        # 파라미터 업데이트
        self.simulation_results['parameters'] = {
            'blood_glucose': self.blood_glucose,
            'standard_concentration': self.standard_concentration,
            'salt_resistance': self.salt_resistance,
            'simulation_duration': self.simulation_duration,
            'sampling_interval': self.sampling_interval,
            'skin_properties': dict(self.skin_properties),
            'environment': dict(self.environment)
        }


# 코드 사용 예시
if __name__ == "__main__":
    # 시뮬레이터 생성
    simulator = GlucoseMonitoringSimulator(
        blood_glucose=130.0,
        standard_concentration=100.0,
        salt_resistance=500.0,
        simulation_duration=300,
        sampling_interval=10
    )
    
    # 피부 특성 설정
    simulator.set_skin_properties(
        thickness=2.5,  # 약간 두꺼운 피부
        permeability=0.4,  # 약간 낮은 투과성
        sweat_rate=1.2  # 보통 이상의 땀 분비
    )
    
    # 환경 조건 설정
    simulator.set_environment(
        temperature=28.0,  # 약간 높은 온도
        humidity=65.0,  # 약간 높은 습도
        ph=6.2  # 약간 산성 땀
    )
    
    # 시뮬레이션 실행
    simulator.run_simulation(real_time=False)
    
    # 보고서 생성
    report_files = simulator.generate_report(output_dir='./simulation_results')
    
    print("\n보고서 생성 완료:")
    for name, path in report_files.items():
        print(f"- {name}: {path}")
