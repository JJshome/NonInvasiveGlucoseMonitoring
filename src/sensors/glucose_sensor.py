"""
포도당 센서 인터페이스 모듈

이 모듈은 비침습적 포도당 측정을 위한 센서 인터페이스를 구현합니다.
제1센서(미지의 포도당 측정)와 제2센서(표준 용액 측정)를 관리하고
데이터 수집 및 처리를 담당합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import threading
import json

class SensorInterface:
    """센서 인터페이스 기본 클래스"""
    
    def __init__(self, sensor_id: str, sampling_rate: float = 1.0):
        """
        초기화 함수
        
        파라미터:
            sensor_id (str): 센서 식별자
            sampling_rate (float): 샘플링 속도 (Hz)
        """
        self.sensor_id = sensor_id
        self.sampling_rate = sampling_rate
        self.is_measuring = False
        self.measurement_thread = None
        self.data_buffer = []
        self.callbacks = []
        
    def start_measurement(self) -> None:
        """측정 시작"""
        if self.is_measuring:
            print(f"센서 {self.sensor_id}는 이미 측정 중입니다.")
            return
            
        self.is_measuring = True
        self.data_buffer = []
        self.measurement_thread = threading.Thread(target=self._measurement_loop)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        
    def stop_measurement(self) -> None:
        """측정 중지"""
        self.is_measuring = False
        if self.measurement_thread:
            self.measurement_thread.join(timeout=2.0)
            
    def _measurement_loop(self) -> None:
        """측정 루프 (내부 메서드)"""
        while self.is_measuring:
            # 센서 읽기
            timestamp = time.time()
            value = self._read_sensor()
            
            # 데이터 저장 및 콜백 호출
            data_point = {
                'timestamp': timestamp,
                'value': value,
                'sensor_id': self.sensor_id
            }
            self.data_buffer.append(data_point)
            
            # 콜백 실행
            for callback in self.callbacks:
                callback(data_point)
                
            # 샘플링 속도에 맞게 대기
            time.sleep(1.0 / self.sampling_rate)
            
    def _read_sensor(self) -> float:
        """
        센서 읽기 (하위 클래스에서 구현 필요)
        
        반환값:
            float: 센서 측정값
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
        
    def get_data(self) -> List[Dict[str, Union[float, str]]]:
        """
        센서 데이터 가져오기
        
        반환값:
            List[Dict[str, Union[float, str]]]: 측정 데이터 목록
        """
        return self.data_buffer
    
    def clear_data(self) -> None:
        """데이터 버퍼 비우기"""
        self.data_buffer = []
        
    def register_callback(self, callback: Callable) -> None:
        """
        데이터 수신 시 호출할 콜백 함수 등록
        
        파라미터:
            callback (Callable): 콜백 함수
        """
        self.callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable) -> None:
        """
        콜백 함수 등록 해제
        
        파라미터:
            callback (Callable): 콜백 함수
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)


class GlucoseSensor(SensorInterface):
    """포도당 측정 센서 클래스"""
    
    def __init__(self, sensor_id: str, 
                 is_standard: bool = False, 
                 standard_concentration: Optional[float] = None,
                 sampling_rate: float = 1.0,
                 noise_level: float = 0.02):
        """
        초기화 함수
        
        파라미터:
            sensor_id (str): 센서 식별자
            is_standard (bool): 표준 센서 여부
            standard_concentration (float, optional): 표준 용액 농도 (mg/dL)
            sampling_rate (float): 샘플링 속도 (Hz)
            noise_level (float): 노이즈 수준 (0~1)
        """
        super().__init__(sensor_id, sampling_rate)
        self.is_standard = is_standard
        self.standard_concentration = standard_concentration
        self.noise_level = noise_level
        
        # 센서 특성 설정
        self.sensitivity = 0.85  # 센서 감도 (0~1)
        self.drift_rate = 0.001  # 센서 드리프트 속도 (%/s)
        self.response_time = 3.0  # 센서 응답 시간 (초)
        
        # 상태 변수
        self.baseline_value = 0.0
        self.current_value = 0.0
        self.start_time = None
        
    def _read_sensor(self) -> float:
        """
        센서 읽기 구현
        
        반환값:
            float: 센서 측정값 (mg/dL)
        """
        if self.start_time is None:
            self.start_time = time.time()
            
            # 초기값 설정
            if self.is_standard and self.standard_concentration is not None:
                self.baseline_value = self.standard_concentration
            else:
                # 실제 센서라면 혈당 범위 내의 무작위 값으로 초기화
                self.baseline_value = np.random.uniform(80, 180)
                
            self.current_value = self.baseline_value
                
        elapsed_time = time.time() - self.start_time
        
        # 표준 센서의 경우 시간에 따라 값이 감소
        if self.is_standard:
            # 시간에 따른 감소 패턴 (지수 감소)
            decay_factor = np.exp(-elapsed_time / (300.0 * self.sensitivity))
            target_value = self.baseline_value * (0.9 * decay_factor + 0.1)
        else:
            # 실제 센서는 피부로부터 포도당 흡수로 값이 증가할 수 있음
            # 실제로는 사용자의 혈당에 따라 달라짐
            target_value = self.baseline_value * (1.0 + 0.05 * (1.0 - np.exp(-elapsed_time / 200.0)))
            
        # 센서 응답 시간 고려
        response_factor = 1.0 - np.exp(-elapsed_time / self.response_time)
        self.current_value = self.current_value + (target_value - self.current_value) * response_factor
        
        # 드리프트 추가
        self.current_value += self.current_value * self.drift_rate * elapsed_time / 1000.0
        
        # 노이즈 추가
        noise = np.random.normal(0, self.noise_level * self.current_value)
        measured_value = self.current_value + noise
        
        return max(0.0, measured_value)  # 음수 방지


class SaltSolutionSensor(SensorInterface):
    """소금물 저항 측정 센서 클래스"""
    
    def __init__(self, sensor_id: str, 
                 initial_resistance: float = 500.0,
                 sampling_rate: float = 1.0,
                 noise_level: float = 0.01):
        """
        초기화 함수
        
        파라미터:
            sensor_id (str): 센서 식별자
            initial_resistance (float): 초기 저항값 (Ω)
            sampling_rate (float): 샘플링 속도 (Hz)
            noise_level (float): 노이즈 수준 (0~1)
        """
        super().__init__(sensor_id, sampling_rate)
        self.initial_resistance = initial_resistance
        self.noise_level = noise_level
        
        # 센서 특성 설정
        self.sensitivity = 0.90  # 센서 감도 (0~1)
        self.drift_rate = 0.0005  # 센서 드리프트 속도 (%/s)
        self.response_time = 2.0  # 센서 응답 시간 (초)
        
        # 상태 변수
        self.baseline_value = initial_resistance
        self.current_value = initial_resistance
        self.start_time = None
        
    def _read_sensor(self) -> float:
        """
        센서 읽기 구현
        
        반환값:
            float: 센서 측정값 (Ω)
        """
        if self.start_time is None:
            self.start_time = time.time()
                
        elapsed_time = time.time() - self.start_time
        
        # 시간에 따른 저항 변화 (처음에는 더 빠르게, 나중에는 느리게 감소)
        decay_factor = np.exp(-elapsed_time / (200.0 * self.sensitivity))
        target_value = self.baseline_value * (0.8 * decay_factor + 0.2)
            
        # 센서 응답 시간 고려
        response_factor = 1.0 - np.exp(-elapsed_time / self.response_time)
        self.current_value = self.current_value + (target_value - self.current_value) * response_factor
        
        # 드리프트 추가
        self.current_value += self.current_value * self.drift_rate * elapsed_time / 1000.0
        
        # 노이즈 추가
        noise = np.random.normal(0, self.noise_level * self.current_value)
        measured_value = self.current_value + noise
        
        return max(0.0, measured_value)  # 음수 방지


class DualSensorSystem:
    """이중 센서 시스템 클래스"""
    
    def __init__(self, standard_concentration: float = 100.0,
                 initial_resistance: float = 500.0):
        """
        초기화 함수
        
        파라미터:
            standard_concentration (float): 표준 포도당 용액 농도 (mg/dL)
            initial_resistance (float): 표준 소금물 초기 저항 (Ω)
        """
        # 센서 생성
        self.primary_sensor = GlucoseSensor(sensor_id="primary", is_standard=False)
        self.standard_glucose_sensor = GlucoseSensor(
            sensor_id="std_glucose", 
            is_standard=True,
            standard_concentration=standard_concentration
        )
        self.salt_sensor = SaltSolutionSensor(
            sensor_id="std_salt",
            initial_resistance=initial_resistance
        )
        
        # 모든 센서 목록
        self.all_sensors = [self.primary_sensor, self.standard_glucose_sensor, self.salt_sensor]
        
        # 측정 결과 저장
        self.measurement_results = {}
        
    def start_all_sensors(self) -> None:
        """모든 센서 측정 시작"""
        for sensor in self.all_sensors:
            sensor.start_measurement()
            
    def stop_all_sensors(self) -> None:
        """모든 센서 측정 중지"""
        for sensor in self.all_sensors:
            sensor.stop_measurement()
            
    def get_all_data(self) -> Dict[str, List[Dict[str, Union[float, str]]]]:
        """
        모든 센서 데이터 가져오기
        
        반환값:
            Dict[str, List[Dict[str, Union[float, str]]]]: 센서별 측정 데이터
        """
        results = {}
        for sensor in self.all_sensors:
            results[sensor.sensor_id] = sensor.get_data()
            
        return results
    
    def get_latest_values(self) -> Dict[str, float]:
        """
        각 센서의 최신 측정값 가져오기
        
        반환값:
            Dict[str, float]: 센서별 최신 측정값
        """
        results = {}
        for sensor in self.all_sensors:
            data = sensor.get_data()
            if data:
                results[sensor.sensor_id] = data[-1]['value']
            else:
                results[sensor.sensor_id] = None
                
        return results
    
    def save_data_to_file(self, filename: str) -> None:
        """
        측정 데이터 파일로 저장
        
        파라미터:
            filename (str): 저장할 파일 경로
        """
        data = self.get_all_data()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_data_from_file(self, filename: str) -> Dict[str, List[Dict[str, Union[float, str]]]]:
        """
        파일에서 측정 데이터 로드
        
        파라미터:
            filename (str): 로드할 파일 경로
            
        반환값:
            Dict[str, List[Dict[str, Union[float, str]]]]: 로드된 데이터
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # 센서 데이터 버퍼 복원
        for sensor in self.all_sensors:
            if sensor.sensor_id in data:
                sensor.data_buffer = data[sensor.sensor_id]
                
        return data


# 코드 사용 예시
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 이중 센서 시스템 생성
    sensor_system = DualSensorSystem(standard_concentration=100.0, initial_resistance=500.0)
    
    # 센서 시작
    sensor_system.start_all_sensors()
    
    # 30초 동안 측정
    print("30초 동안 센서 측정 중...")
    for i in range(30):
        time.sleep(1)
        
        # 최신 값 출력
        latest_values = sensor_system.get_latest_values()
        print(f"[{i+1}초] 측정값: "
              f"기본 센서={latest_values['primary']:.1f} mg/dL, "
              f"표준 포도당={latest_values['std_glucose']:.1f} mg/dL, "
              f"표준 소금물={latest_values['std_salt']:.1f} Ω")
    
    # 센서 중지
    sensor_system.stop_all_sensors()
    
    # 결과 데이터 가져오기
    all_data = sensor_system.get_all_data()
    
    # 그래프 그리기
    plt.figure(figsize=(12, 8))
    
    # 포도당 센서 그래프
    plt.subplot(2, 1, 1)
    
    # 기본 센서 데이터
    primary_values = [d['value'] for d in all_data['primary']]
    primary_times = [i for i in range(len(primary_values))]
    plt.plot(primary_times, primary_values, 'b-', label='기본 센서 (mg/dL)')
    
    # 표준 포도당 센서 데이터
    std_glucose_values = [d['value'] for d in all_data['std_glucose']]
    std_glucose_times = [i for i in range(len(std_glucose_values))]
    plt.plot(std_glucose_times, std_glucose_values, 'r--', label='표준 포도당 센서 (mg/dL)')
    
    plt.xlabel('시간 (초)')
    plt.ylabel('포도당 (mg/dL)')
    plt.legend()
    plt.title('포도당 센서 측정값')
    plt.grid(True)
    
    # 소금물 센서 그래프
    plt.subplot(2, 1, 2)
    
    # 소금물 센서 데이터
    salt_values = [d['value'] for d in all_data['std_salt']]
    salt_times = [i for i in range(len(salt_values))]
    plt.plot(salt_times, salt_values, 'g-', label='표준 소금물 센서 (Ω)')
    
    plt.xlabel('시간 (초)')
    plt.ylabel('저항 (Ω)')
    plt.legend()
    plt.title('소금물 센서 측정값')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensor_data.png')
    
    # 데이터 저장
    sensor_system.save_data_to_file('sensor_data.json')
    
    print("측정 완료. 결과 파일 저장됨: sensor_data.json, sensor_data.png")
