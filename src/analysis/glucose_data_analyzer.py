"""
포도당 측정 데이터 분석 모듈

이 모듈은 표준 용액 기반 비침습적 포도당 측정 데이터를 분석하는 기능을 제공합니다.
시계열 데이터 분석, 통계적 처리, 보정 효과 평가 등의 기능을 포함합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from datetime import datetime


class GlucoseDataAnalyzer:
    """포도당 측정 데이터 분석 클래스"""
    
    def __init__(self):
        """초기화 함수"""
        self.measurement_data = None
        self.reference_data = None
        self.results = {}
        
    def load_data_from_file(self, filepath: str) -> None:
        """
        파일에서 데이터 로드
        
        파라미터:
            filepath (str): 데이터 파일 경로
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.measurement_data = data
        print(f"{filepath}에서 데이터 로드 완료")
        
    def load_reference_data(self, filepath: str) -> None:
        """
        참조 데이터 로드 (실제 혈당 측정값 등)
        
        파라미터:
            filepath (str): 참조 데이터 파일 경로
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.reference_data = data
        print(f"{filepath}에서 참조 데이터 로드 완료")
        
    def prepare_time_series_data(self, 
                               sensor_id: str, 
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> pd.DataFrame:
        """
        센서 데이터를 시계열 형식으로 변환
        
        파라미터:
            sensor_id (str): 센서 ID
            start_time (float, optional): 시작 시간 (타임스탬프)
            end_time (float, optional): 종료 시간 (타임스탬프)
            
        반환값:
            pd.DataFrame: 시계열 형식의 데이터프레임
        """
        if self.measurement_data is None or sensor_id not in self.measurement_data:
            raise ValueError(f"센서 {sensor_id}의 데이터가 없습니다.")
            
        # 해당 센서 데이터 추출
        sensor_data = self.measurement_data[sensor_id]
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(sensor_data)
        
        # 타임스탬프를 인덱스로 설정
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # 시간 범위 필터링
        if start_time is not None:
            start_datetime = pd.to_datetime(start_time, unit='s')
            df = df[df.index >= start_datetime]
            
        if end_time is not None:
            end_datetime = pd.to_datetime(end_time, unit='s')
            df = df[df.index <= end_datetime]
            
        return df
    
    def calculate_statistics(self, 
                           sensor_id: str, 
                           time_window: Optional[int] = None) -> Dict[str, float]:
        """
        센서 데이터의 통계 정보 계산
        
        파라미터:
            sensor_id (str): 센서 ID
            time_window (int, optional): 시간 윈도우 (초)
            
        반환값:
            Dict[str, float]: 통계 정보
        """
        df = self.prepare_time_series_data(sensor_id)
        
        # 마지막 time_window 초 동안의 데이터만 사용
        if time_window is not None:
            last_time = df.index.max()
            start_time = last_time - pd.Timedelta(seconds=time_window)
            df = df[df.index >= start_time]
            
        # 통계량 계산
        stats_dict = {
            'mean': df['value'].mean(),
            'std': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max(),
            'median': df['value'].median(),
            'count': df['value'].count(),
            'last_value': df['value'].iloc[-1] if not df.empty else None
        }
        
        # 센서별 적절한 단위 표시
        if sensor_id.startswith('std_salt'):
            stats_dict['unit'] = 'Ω'
        else:
            stats_dict['unit'] = 'mg/dL'
            
        return stats_dict
    
    def calculate_diffusion_rate(self, 
                               standard_sensor_id: str,
                               window_size: int = 10) -> Tuple[float, float]:
        """
        표준 센서의 확산율 계산
        
        파라미터:
            standard_sensor_id (str): 표준 센서 ID
            window_size (int): 이동 평균 윈도우 크기
            
        반환값:
            Tuple[float, float]: (확산율, R² 값)
        """
        df = self.prepare_time_series_data(standard_sensor_id)
        
        # 시계열 데이터 준비
        df['elapsed_seconds'] = (df.index - df.index.min()).total_seconds()
        
        # 이동 평균으로 노이즈 감소
        if len(df) >= window_size:
            df['smoothed_value'] = df['value'].rolling(window=window_size).mean()
            df = df.dropna()
        else:
            df['smoothed_value'] = df['value']
        
        # 선형 회귀로 확산율 계산
        x = df['elapsed_seconds'].values
        y = df['smoothed_value'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope, r_value**2
    
    def compare_calibrated_vs_uncalibrated(self, 
                                         primary_sensor_id: str,
                                         standard_sensor_id: str,
                                         reference_value: Optional[float] = None) -> Dict[str, Any]:
        """
        보정 전후 정확도 비교
        
        파라미터:
            primary_sensor_id (str): 주 센서 ID
            standard_sensor_id (str): 표준 센서 ID
            reference_value (float, optional): 참조값 (혈당계 측정값 등)
            
        반환값:
            Dict[str, Any]: 비교 결과
        """
        # 측정 데이터 준비
        primary_stats = self.calculate_statistics(primary_sensor_id, time_window=60)
        std_stats = self.calculate_statistics(standard_sensor_id, time_window=60)
        
        # 확산율 계산
        diffusion_rate, r_squared = self.calculate_diffusion_rate(standard_sensor_id)
        
        # 초기 농도와 현재 농도 차이에 기반한 보정 계수 계산
        initial_value = self.prepare_time_series_data(standard_sensor_id)['value'].iloc[0]
        current_value = std_stats['last_value']
        
        # 보정 계수 계산
        if initial_value is not None and current_value is not None:
            concentration_change_ratio = (initial_value - current_value) / initial_value
            calibration_factor = 1.0 + concentration_change_ratio
        else:
            calibration_factor = 1.0
            
        # 보정된 측정값
        uncalibrated_value = primary_stats['last_value']
        calibrated_value = uncalibrated_value * calibration_factor
        
        # 참조값이 있으면 오차 계산
        error_uncalibrated = None
        error_calibrated = None
        improvement = None
        
        if reference_value is not None:
            error_uncalibrated = (uncalibrated_value - reference_value) / reference_value * 100
            error_calibrated = (calibrated_value - reference_value) / reference_value * 100
            
            # 개선도 계산 (오차 감소율)
            if abs(error_uncalibrated) > 0:
                improvement = (abs(error_uncalibrated) - abs(error_calibrated)) / abs(error_uncalibrated) * 100
            else:
                improvement = 0.0
                
        # 결과 사전
        result = {
            'uncalibrated_value': uncalibrated_value,
            'calibrated_value': calibrated_value,
            'calibration_factor': calibration_factor,
            'diffusion_rate': diffusion_rate,
            'r_squared': r_squared,
            'reference_value': reference_value,
            'error_uncalibrated': error_uncalibrated,
            'error_calibrated': error_calibrated,
            'improvement': improvement,
            'unit': primary_stats['unit']
        }
        
        return result
    
    def analyze_multiple_samples(self, 
                              samples: List[Dict[str, Any]],
                              reference_values: List[float]) -> pd.DataFrame:
        """
        여러 샘플의 보정 전후 결과 분석
        
        파라미터:
            samples (List[Dict[str, Any]]): 샘플 데이터 목록
            reference_values (List[float]): 참조값 목록
            
        반환값:
            pd.DataFrame: 분석 결과 데이터프레임
        """
        results = []
        
        for i, (sample, ref_value) in enumerate(zip(samples, reference_values)):
            # 샘플 분석
            result = self.compare_calibrated_vs_uncalibrated(
                primary_sensor_id=sample['primary_sensor_id'],
                standard_sensor_id=sample['standard_sensor_id'],
                reference_value=ref_value
            )
            
            # 샘플 ID 추가
            result['sample_id'] = i + 1
            
            results.append(result)
            
        # 데이터프레임으로 변환
        df = pd.DataFrame(results)
        
        # 요약 통계 계산
        summary = {
            'mean_error_uncalibrated': df['error_uncalibrated'].abs().mean(),
            'mean_error_calibrated': df['error_calibrated'].abs().mean(),
            'max_error_uncalibrated': df['error_uncalibrated'].abs().max(),
            'max_error_calibrated': df['error_calibrated'].abs().max(),
            'mean_improvement': df['improvement'].mean(),
            'sample_count': len(df)
        }
        
        # 결과 저장
        self.results['multiple_samples'] = {
            'details': df.to_dict('records'),
            'summary': summary
        }
        
        return df
    
    def plot_time_series(self, 
                       sensor_ids: List[str],
                       title: str = "센서 측정값 시계열",
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        센서 데이터 시계열 그래프
        
        파라미터:
            sensor_ids (List[str]): 센서 ID 목록
            title (str): 그래프 제목
            figsize (Tuple[int, int]): 그래프 크기
            
        반환값:
            plt.Figure: 생성된 그래프
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_ids)))
        
        for i, sensor_id in enumerate(sensor_ids):
            df = self.prepare_time_series_data(sensor_id)
            
            # 시간 축 정규화 (첫 번째 측정부터 경과 시간)
            df['elapsed_seconds'] = (df.index - df.index.min()).total_seconds()
            
            # 플롯 스타일 결정
            if 'primary' in sensor_id:
                style = '-'
                label = '제1센서 (미지의 포도당)'
            elif 'std_glucose' in sensor_id:
                style = '--'
                label = '제2센서 (표준 포도당 용액)'
            elif 'std_salt' in sensor_id:
                style = '-.'
                label = '제2센서 (표준 소금물)'
            else:
                style = ':'
                label = sensor_id
                
            ax.plot(df['elapsed_seconds'], df['value'], style, color=colors[i], label=label)
            
        # 그래프 설정
        ax.set_xlabel('시간 (초)')
        
        # Y축 레이블 설정 (센서 타입에 따라)
        if all('salt' in s for s in sensor_ids):
            ax.set_ylabel('저항 (Ω)')
        elif all('salt' not in s for s in sensor_ids):
            ax.set_ylabel('포도당 농도 (mg/dL)')
        else:
            ax.set_ylabel('측정값')
            
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def plot_calibration_comparison(self, 
                                 result: Dict[str, Any],
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        보정 전후 비교 그래프
        
        파라미터:
            result (Dict[str, Any]): compare_calibrated_vs_uncalibrated 결과
            figsize (Tuple[int, int]): 그래프 크기
            
        반환값:
            plt.Figure: 생성된 그래프
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 데이터 준비
        values = [
            result['uncalibrated_value'],
            result['calibrated_value'],
            result['reference_value'] if result['reference_value'] is not None else 0
        ]
        
        labels = ['보정 전', '보정 후', '참조값']
        colors = ['#ff6b6b', '#4dabf7', '#20c997']
        
        # 참조값이 없으면 제외
        if result['reference_value'] is None:
            values = values[:2]
            labels = labels[:2]
            colors = colors[:2]
            
        # 바 차트
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}',
                   ha='center', va='bottom')
            
        # 오차 정보 추가
        if result['reference_value'] is not None:
            error_text = (
                f"보정 전 오차: {result['error_uncalibrated']:.2f}%\n"
                f"보정 후 오차: {result['error_calibrated']:.2f}%\n"
                f"개선율: {result['improvement']:.2f}%"
            )
            ax.text(0.95, 0.05, error_text,
                   transform=ax.transAxes,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        # 그래프 설정
        ax.set_ylabel(f"측정값 ({result['unit']})")
        ax.set_title("보정 전후 측정값 비교")
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        
        return fig
    
    def save_results_to_file(self, filepath: str) -> None:
        """
        분석 결과 파일로 저장
        
        파라미터:
            filepath (str): 저장할 파일 경로
        """
        # 측정 시간 추가
        self.results['analysis_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"분석 결과가 {filepath}에 저장되었습니다.")


# 코드 사용 예시
if __name__ == "__main__":
    # 분석기 생성
    analyzer = GlucoseDataAnalyzer()
    
    # 실제 데이터 파일이 있다면 로드
    try:
        analyzer.load_data_from_file('sensor_data.json')
    except FileNotFoundError:
        print("센서 데이터 파일을 찾을 수 없습니다. 샘플 데이터를 생성합니다.")
        
        # 샘플 데이터 생성
        sample_data = {
            "primary": [
                {"timestamp": 1621500000 + i*30, "value": 120 + i*0.5 + np.random.normal(0, 2), "sensor_id": "primary"}
                for i in range(20)
            ],
            "std_glucose": [
                {"timestamp": 1621500000 + i*30, "value": 100 - i*0.3 + np.random.normal(0, 1), "sensor_id": "std_glucose"}
                for i in range(20)
            ],
            "std_salt": [
                {"timestamp": 1621500000 + i*30, "value": 500 - i*2 + np.random.normal(0, 5), "sensor_id": "std_salt"}
                for i in range(20)
            ]
        }
        
        analyzer.measurement_data = sample_data
        
    # 시계열 그래프
    fig1 = analyzer.plot_time_series(
        sensor_ids=["primary", "std_glucose"],
        title="포도당 센서 측정값 시계열"
    )
    fig1.savefig('glucose_time_series.png')
    
    fig2 = analyzer.plot_time_series(
        sensor_ids=["std_salt"],
        title="소금물 센서 측정값 시계열"
    )
    fig2.savefig('salt_time_series.png')
    
    # 보정 전후 비교
    result = analyzer.compare_calibrated_vs_uncalibrated(
        primary_sensor_id="primary",
        standard_sensor_id="std_glucose",
        reference_value=135  # 예상 혈당값
    )
    
    print("\n보정 결과:")
    print(f"보정 전 측정값: {result['uncalibrated_value']:.1f} mg/dL")
    print(f"보정 후 측정값: {result['calibrated_value']:.1f} mg/dL")
    print(f"보정 계수: {result['calibration_factor']:.4f}")
    print(f"확산율: {result['diffusion_rate']:.6f}")
    
    if result['reference_value'] is not None:
        print(f"참조값: {result['reference_value']:.1f} mg/dL")
        print(f"보정 전 오차: {result['error_uncalibrated']:.2f}%")
        print(f"보정 후 오차: {result['error_calibrated']:.2f}%")
        print(f"개선율: {result['improvement']:.2f}%")
    
    # 보정 비교 그래프
    fig3 = analyzer.plot_calibration_comparison(result)
    fig3.savefig('calibration_comparison.png')
    
    # 결과 저장
    analyzer.results['single_sample'] = result
    analyzer.save_results_to_file('analysis_results.json')
    
    print("\n분석 완료. 그래프 파일 저장됨: glucose_time_series.png, salt_time_series.png, calibration_comparison.png")
