"""
시뮬레이션 유틸리티 함수 모듈

이 모듈은 표준 용액 기반 비침습적 체외 진단 시뮬레이션에 사용되는
다양한 유틸리티 함수들을 제공합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: str = '.') -> Dict[str, str]:
    """
    필요한 디렉토리 생성
    
    파라미터:
        base_dir (str): 기본 디렉토리 경로
        
    반환값:
        Dict[str, str]: 생성된 디렉토리 경로 사전
    """
    dirs = {
        'results': os.path.join(base_dir, 'results'),
        'cache': os.path.join(base_dir, 'cache'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"디렉토리 생성됨: {path}")
        
    return dirs


def generate_timestamp_filename(prefix: str, extension: str = 'json') -> str:
    """
    타임스탬프가 포함된 파일명 생성
    
    파라미터:
        prefix (str): 파일명 접두사
        extension (str): 파일 확장자
        
    반환값:
        str: 생성된 파일명
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def save_json_data(data: Dict[str, Any], filepath: str) -> None:
    """
    데이터를 JSON 파일로 저장
    
    파라미터:
        data (Dict[str, Any]): 저장할 데이터
        filepath (str): 저장 경로
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"데이터가 저장됨: {filepath}")


def load_json_data(filepath: str) -> Dict[str, Any]:
    """
    JSON 파일에서 데이터 로드
    
    파라미터:
        filepath (str): 로드할 파일 경로
        
    반환값:
        Dict[str, Any]: 로드된 데이터
    """
    if not os.path.exists(filepath):
        logger.error(f"파일을 찾을 수 없음: {filepath}")
        return {}
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"데이터가 로드됨: {filepath}")
    return data


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    값 목록의 통계 계산
    
    파라미터:
        values (List[float]): 값 목록
        
    반환값:
        Dict[str, float]: 통계 정보
    """
    if not values:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'count': 0
        }
        
    values_array = np.array(values)
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'count': len(values)
    }


def smooth_data(values: List[float], window_size: int = 5) -> List[float]:
    """
    이동 평균으로 데이터 스무딩
    
    파라미터:
        values (List[float]): 원본 데이터
        window_size (int): 윈도우 크기
        
    반환값:
        List[float]: 스무딩된 데이터
    """
    if len(values) < window_size:
        return values
        
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
        
    return smoothed


def fit_linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    선형 회귀 계산
    
    파라미터:
        x (List[float]): x 값 목록
        y (List[float]): y 값 목록
        
    반환값:
        Tuple[float, float, float]: (기울기, y절편, R²)
    """
    if len(x) != len(y) or len(x) < 2:
        logger.error("선형 회귀에 유효하지 않은 데이터")
        return 0.0, 0.0, 0.0
        
    x_array = np.array(x)
    y_array = np.array(y)
    
    # 선형 회귀 계산
    n = len(x)
    x_mean = np.mean(x_array)
    y_mean = np.mean(y_array)
    
    numerator = np.sum((x_array - x_mean) * (y_array - y_mean))
    denominator = np.sum((x_array - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0, 0.0, 0.0
        
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # R² 계산
    y_pred = slope * x_array + intercept
    ss_total = np.sum((y_array - y_mean) ** 2)
    ss_residual = np.sum((y_array - y_pred) ** 2)
    
    if ss_total == 0:
        r_squared = 0
    else:
        r_squared = 1 - (ss_residual / ss_total)
        
    return slope, intercept, r_squared


def generate_calibration_plot(
    uncalibrated: float,
    calibrated: float,
    reference: Optional[float] = None,
    title: str = "보정 전후 측정값 비교",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    보정 전후 비교 그래프 생성
    
    파라미터:
        uncalibrated (float): 보정 전 값
        calibrated (float): 보정 후 값
        reference (float, optional): 참조값
        title (str): 그래프 제목
        figsize (Tuple[int, int]): 그래프 크기
        
    반환값:
        plt.Figure: 생성된 그래프
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 데이터 준비
    labels = ['보정 전', '보정 후']
    values = [uncalibrated, calibrated]
    colors = ['#ff6b6b', '#4dabf7']
    
    if reference is not None:
        labels.append('참조값')
        values.append(reference)
        colors.append('#20c997')
        
    # 바 차트
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}',
               ha='center', va='bottom')
        
    # 그래프 설정
    ax.set_ylabel("측정값 (mg/dL)")
    ax.set_title(title)
    ax.grid(True, axis='y')
    
    # 오차 정보 추가
    if reference is not None:
        error_uncalibrated = (uncalibrated - reference) / reference * 100
        error_calibrated = (calibrated - reference) / reference * 100
        
        if abs(error_uncalibrated) > 0:
            improvement = (abs(error_uncalibrated) - abs(error_calibrated)) / abs(error_uncalibrated) * 100
        else:
            improvement = 0.0
            
        error_text = (
            f"보정 전 오차: {error_uncalibrated:.2f}%\n"
            f"보정 후 오차: {error_calibrated:.2f}%\n"
            f"개선율: {improvement:.2f}%"
        )
        ax.text(0.95, 0.05, error_text,
               transform=ax.transAxes,
               ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    return fig


def generate_time_series_plot(
    timestamps: List[float],
    values: List[float],
    title: str = "시간에 따른 측정값",
    xlabel: str = "시간 (초)",
    ylabel: str = "측정값",
    color: str = '#0066cc',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    시계열 그래프 생성
    
    파라미터:
        timestamps (List[float]): 시간 값 목록
        values (List[float]): 측정값 목록
        title (str): 그래프 제목
        xlabel (str): x축 레이블
        ylabel (str): y축 레이블
        color (str): 그래프 색상
        figsize (Tuple[int, int]): 그래프 크기
        
    반환값:
        plt.Figure: 생성된 그래프
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 시간 정규화 (첫 번째 타임스탬프 기준)
    if timestamps:
        normalized_time = [(t - timestamps[0]) for t in timestamps]
    else:
        normalized_time = []
        
    # 그래프 그리기
    ax.plot(normalized_time, values, '-', color=color, linewidth=2)
    
    # 그래프 설정
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def format_duration(seconds: int) -> str:
    """
    시간을 가독성 있는 형식으로 변환
    
    파라미터:
        seconds (int): 초 단위 시간
        
    반환값:
        str: 형식화된 시간 문자열
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"