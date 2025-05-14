"""
표준 용액 기반 비침습적 체외 진단 시뮬레이션 웹 인터페이스

이 모듈은 Flask 기반 웹 인터페이스를 제공하여 사용자가 쉽게 시뮬레이션을
실행하고 결과를 확인할 수 있도록 합니다.

작성자: JJshome
날짜: 2025-05-14
버전: 1.0.0
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
import time
import uuid
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import numpy as np
import threading
import sys

# 상위 디렉토리 모듈 로드를 위한 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 시뮬레이터 모듈 임포트
from deployment.simulation.simulator import GlucoseMonitoringSimulator

# Flask 앱 초기화
app = Flask(__name__, static_folder='static', template_folder='templates')

# 결과 저장 디렉토리
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 진행 중인 시뮬레이션 관리
active_simulations = {}


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """시뮬레이션 실행 API"""
    try:
        # 폼 데이터 가져오기
        data = request.json
        
        # 기본 파라미터
        blood_glucose = float(data.get('blood_glucose', 120.0))
        standard_concentration = float(data.get('standard_concentration', 100.0))
        salt_resistance = float(data.get('salt_resistance', 500.0))
        simulation_duration = int(data.get('simulation_duration', 300))
        
        # 피부 특성
        skin_thickness = float(data.get('skin_thickness', 2.0))
        skin_permeability = float(data.get('skin_permeability', 0.5))
        sweat_rate = float(data.get('sweat_rate', 1.0))
        
        # 환경 조건
        temperature = float(data.get('temperature', 25.0))
        humidity = float(data.get('humidity', 50.0))
        ph = float(data.get('ph', 6.5))
        
        # 시뮬레이션 ID 생성
        sim_id = str(uuid.uuid4())
        
        # 시뮬레이터 초기화
        simulator = GlucoseMonitoringSimulator(
            blood_glucose=blood_glucose,
            standard_concentration=standard_concentration,
            salt_resistance=salt_resistance,
            simulation_duration=simulation_duration,
            sampling_interval=10
        )
        
        # 피부 특성 설정
        simulator.set_skin_properties(
            thickness=skin_thickness,
            permeability=skin_permeability,
            sweat_rate=sweat_rate
        )
        
        # 환경 조건 설정
        simulator.set_environment(
            temperature=temperature,
            humidity=humidity,
            ph=ph
        )
        
        # 시뮬레이션 정보 저장
        active_simulations[sim_id] = {
            'simulator': simulator,
            'status': 'initialized',
            'progress': 0,
            'start_time': time.time(),
            'results': None
        }
        
        # 백그라운드에서 시뮬레이션 실행
        threading.Thread(target=_run_simulation_thread, args=(sim_id,)).start()
        
        return jsonify({
            'success': True,
            'simulation_id': sim_id,
            'message': '시뮬레이션이 시작되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류 발생: {str(e)}'
        }), 500


def _run_simulation_thread(sim_id):
    """백그라운드 시뮬레이션 스레드"""
    try:
        sim_info = active_simulations[sim_id]
        simulator = sim_info['simulator']
        
        # 상태 업데이트
        sim_info['status'] = 'running'
        
        # 시뮬레이션 시작 시간
        start_time = time.time()
        
        # 진행 상황 업데이트 함수
        def update_progress(elapsed, duration):
            progress = min(100, int(elapsed / duration * 100))
            sim_info['progress'] = progress
        
        # 시뮬레이션 실행
        simulation_duration = simulator.simulation_duration
        
        # 백그라운드에서 실행하면서 주기적으로 진행 상황 업데이트
        simulation_thread = threading.Thread(
            target=lambda: simulator.run_simulation(real_time=False)
        )
        simulation_thread.start()
        
        while simulation_thread.is_alive():
            elapsed = time.time() - start_time
            update_progress(elapsed, simulation_duration)
            time.sleep(0.5)
            
        simulation_thread.join()
        
        # 결과 생성
        result_files = simulator.generate_report(output_dir=RESULTS_DIR)
        
        # 결과 기록
        sim_info['status'] = 'completed'
        sim_info['progress'] = 100
        sim_info['results'] = {
            'files': result_files,
            'summary': {
                'blood_glucose': simulator.blood_glucose,
                'uncalibrated_value': simulator.simulation_results['analysis_results']['glucose_calibration']['uncalibrated_value'],
                'calibrated_value': simulator.simulation_results['analysis_results']['glucose_calibration']['calibrated_value'],
                'error_uncalibrated': simulator.simulation_results['analysis_results']['glucose_calibration']['error_uncalibrated'],
                'error_calibrated': simulator.simulation_results['analysis_results']['glucose_calibration']['error_calibrated'],
                'improvement': simulator.simulation_results['analysis_results']['glucose_calibration']['improvement']
            }
        }
        
    except Exception as e:
        if sim_id in active_simulations:
            active_simulations[sim_id]['status'] = 'error'
            active_simulations[sim_id]['error_message'] = str(e)
        print(f"시뮬레이션 오류: {str(e)}")


@app.route('/simulation_status/<sim_id>', methods=['GET'])
def simulation_status(sim_id):
    """시뮬레이션 상태 확인 API"""
    if sim_id not in active_simulations:
        return jsonify({
            'success': False,
            'message': '시뮬레이션을 찾을 수 없습니다.'
        }), 404
        
    sim_info = active_simulations[sim_id]
    
    response = {
        'success': True,
        'status': sim_info['status'],
        'progress': sim_info['progress']
    }
    
    if sim_info['status'] == 'completed':
        response['results'] = sim_info['results']
    elif sim_info['status'] == 'error':
        response['error_message'] = sim_info.get('error_message', '알 수 없는 오류')
        
    return jsonify(response)


@app.route('/results/<path:filename>')
def download_result(filename):
    """결과 파일 다운로드"""
    return send_from_directory(RESULTS_DIR, filename)


@app.route('/view_result/<sim_id>')
def view_result(sim_id):
    """시뮬레이션 결과 페이지"""
    if sim_id not in active_simulations or active_simulations[sim_id]['status'] != 'completed':
        return redirect(url_for('index'))
        
    return render_template(
        'result.html',
        sim_id=sim_id,
        results=active_simulations[sim_id]['results']
    )


@app.route('/templates/<path:template_name>')
def get_template(template_name):
    """HTML 템플릿 제공"""
    return render_template(template_name)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
