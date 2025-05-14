/**
 * 시뮬레이션 UI 관련 JavaScript 함수들
 */

// DOM이 로드된 후 실행
document.addEventListener('DOMContentLoaded', function() {
    initializeFormControls();
    setupEventListeners();
});

/**
 * 폼 컨트롤 초기화
 */
function initializeFormControls() {
    // 범위 슬라이더 값 표시 초기화
    updateRangeDisplays();
    
    // 유효성 검사 설정
    setupValidation();
}

/**
 * 이벤트 리스너 설정
 */
function setupEventListeners() {
    // 범위 슬라이더 이벤트 리스너
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        input.addEventListener('input', updateRangeDisplays);
    });
    
    // 시뮬레이션 실행 버튼
    const runButton = document.getElementById('runSimulation');
    if (runButton) {
        runButton.addEventListener('click', runSimulation);
    }
}

/**
 * 범위 슬라이더 표시값 업데이트
 */
function updateRangeDisplays() {
    // 시뮬레이션 시간
    const durationSlider = document.getElementById('simulation_duration');
    const durationDisplay = document.getElementById('duration_display');
    if (durationSlider && durationDisplay) {
        durationDisplay.textContent = durationSlider.value + '초';
    }
    
    // 피부 두께
    const thicknessSlider = document.getElementById('skin_thickness');
    const thicknessDisplay = document.getElementById('thickness_display');
    if (thicknessSlider && thicknessDisplay) {
        thicknessDisplay.textContent = thicknessSlider.value + 'mm';
    }
    
    // 피부 투과성
    const permeabilitySlider = document.getElementById('skin_permeability');
    const permeabilityDisplay = document.getElementById('permeability_display');
    if (permeabilitySlider && permeabilityDisplay) {
        permeabilityDisplay.textContent = permeabilitySlider.value;
    }
    
    // 땀 분비 속도
    const sweatSlider = document.getElementById('sweat_rate');
    const sweatDisplay = document.getElementById('sweat_display');
    if (sweatSlider && sweatDisplay) {
        sweatDisplay.textContent = sweatSlider.value;
    }
    
    // 온도
    const tempSlider = document.getElementById('temperature');
    const tempDisplay = document.getElementById('temp_display');
    if (tempSlider && tempDisplay) {
        tempDisplay.textContent = tempSlider.value + '°C';
    }
    
    // 습도
    const humiditySlider = document.getElementById('humidity');
    const humidityDisplay = document.getElementById('humidity_display');
    if (humiditySlider && humidityDisplay) {
        humidityDisplay.textContent = humiditySlider.value + '%';
    }
    
    // pH
    const phSlider = document.getElementById('ph');
    const phDisplay = document.getElementById('ph_display');
    if (phSlider && phDisplay) {
        phDisplay.textContent = phSlider.value;
    }
}

/**
 * 폼 유효성 검사 설정
 */
function setupValidation() {
    // 숫자 입력 필드 검사
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.min || '-Infinity');
            const max = parseFloat(this.max || 'Infinity');
            let value = parseFloat(this.value);
            
            if (isNaN(value)) {
                this.value = '';
            } else if (value < min) {
                this.value = min;
            } else if (value > max) {
                this.value = max;
            }
        });
    });
}

/**
 * 시뮬레이션 실행
 */
function runSimulation() {
    // 폼 데이터 수집
    const formData = {
        blood_glucose: parseFloat(document.getElementById('blood_glucose').value),
        standard_concentration: parseFloat(document.getElementById('standard_concentration').value),
        salt_resistance: parseFloat(document.getElementById('salt_resistance').value),
        simulation_duration: parseInt(document.getElementById('simulation_duration').value),
        skin_thickness: parseFloat(document.getElementById('skin_thickness').value),
        skin_permeability: parseFloat(document.getElementById('skin_permeability').value),
        sweat_rate: parseFloat(document.getElementById('sweat_rate').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value)
    };
    
    // 유효성 검사
    if (!validateSimulationData(formData)) {
        return;
    }
    
    // 로딩 오버레이 표시
    showLoadingOverlay('시뮬레이션 초기화 중...');
    
    // API 호출
    fetch('/run_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 시뮬레이션 ID를 사용하여 상태 확인 시작
            checkSimulationStatus(data.simulation_id);
        } else {
            hideLoadingOverlay();
            showError('시뮬레이션 시작 오류: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoadingOverlay();
        showError('시뮬레이션 요청 중 오류가 발생했습니다.');
    });
}

/**
 * 시뮬레이션 데이터 유효성 검사
 */
function validateSimulationData(data) {
    if (isNaN(data.blood_glucose) || data.blood_glucose < 50 || data.blood_glucose > 500) {
        showError('혈당값은 50~500 mg/dL 범위여야 합니다.');
        return false;
    }
    
    if (isNaN(data.standard_concentration) || data.standard_concentration < 50 || data.standard_concentration > 300) {
        showError('표준 포도당 농도는 50~300 mg/dL 범위여야 합니다.');
        return false;
    }
    
    if (isNaN(data.salt_resistance) || data.salt_resistance < 100 || data.salt_resistance > 1000) {
        showError('표준 소금물 저항은 100~1000 Ω 범위여야 합니다.');
        return false;
    }
    
    return true;
}

/**
 * 시뮬레이션 상태 확인
 */
function checkSimulationStatus(simId) {
    fetch(`/simulation_status/${simId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 진행 상황 업데이트
                updateProgressBar(data.progress);
                
                if (data.status === 'completed') {
                    updateStatusMessage('시뮬레이션 완료!');
                    setTimeout(() => {
                        window.location.href = `/view_result/${simId}`;
                    }, 1000);
                } else if (data.status === 'error') {
                    updateStatusMessage('오류 발생: ' + data.error_message);
                    setTimeout(() => {
                        hideLoadingOverlay();
                        showError('시뮬레이션 오류: ' + data.error_message);
                    }, 2000);
                } else {
                    // 진행 중인 경우 계속 확인
                    updateStatusMessage(`진행 중... ${data.progress}%`);
                    setTimeout(() => checkSimulationStatus(simId), 1000);
                }
            } else {
                hideLoadingOverlay();
                showError('상태 확인 오류: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            hideLoadingOverlay();
            showError('상태 확인 중 오류가 발생했습니다.');
        });
}

/**
 * 로딩 오버레이 표시
 */
function showLoadingOverlay(message) {
    const overlay = document.getElementById('loadingOverlay');
    const statusMessage = document.getElementById('statusMessage');
    
    if (overlay) {
        overlay.style.display = 'flex';
    }
    
    if (statusMessage && message) {
        statusMessage.textContent = message;
    }
    
    // 진행 바 초기화
    updateProgressBar(0);
}

/**
 * 로딩 오버레이 숨기기
 */
function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * 진행 바 업데이트
 */
function updateProgressBar(progress) {
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = progress + '%';
    }
}

/**
 * 상태 메시지 업데이트
 */
function updateStatusMessage(message) {
    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.textContent = message;
    }
}

/**
 * 오류 메시지 표시
 */
function showError(message) {
    alert(message);
}

/**
 * 결과 페이지에서의 오차 클래스 설정
 */
function setupErrorClasses() {
    const uncalibratedError = document.getElementById('uncalibrated_error');
    const calibratedError = document.getElementById('calibrated_error');
    
    if (uncalibratedError) {
        // 보정 전 오차 클래스 설정
        const uncalibErrorVal = parseFloat(uncalibratedError.textContent);
        if (Math.abs(uncalibErrorVal) <= 5) {
            uncalibratedError.className = 'positive';
        } else if (Math.abs(uncalibErrorVal) <= 10) {
            uncalibratedError.className = '';
        } else {
            uncalibratedError.className = 'negative';
        }
    }
    
    if (calibratedError) {
        // 보정 후 오차 클래스 설정
        const calibErrorVal = parseFloat(calibratedError.textContent);
        if (Math.abs(calibErrorVal) <= 5) {
            calibratedError.className = 'positive';
        } else if (Math.abs(calibErrorVal) <= 10) {
            calibratedError.className = '';
        } else {
            calibratedError.className = 'negative';
        }
    }
}