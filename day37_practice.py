# day37_practice.py
# 주제: 캠퍼스 시설 안내 음성 도우미

# [실습 목표]
# 녹음 → STT → 응답 생성 → TTS → 재생' 흐름을 하나의 실제 서비스 시나리오로 확장
# Git Repository 생성
# Render에 코드 배포
# Neon에 데이터베이스 생성
# End-point에서 테스트

# [완성 화면의 동작]
# Streamlit에서 마이크로 음성을 녹음합니다.
# 데모 STT 또는 실제 faster-whisper를 선택합니다.
# 안내할 시설(도서관, 학생식당, 취업지원센터, 행정실)에 대한
# 한국어 질문을 인식합니다.
# 사용자의 질문에서 시설명을 찾아 운영 안내를 반환합니다.
# 생성한 답변을 비프음 데모 또는 실제 edge-tts로 출력합니다.
# STT/응답/TTS/총 처리시간을 측정하고 막대그래프를 표시합니다.
# 개인정보를 제외한 실행 이력을 JSONL에 저장합니다.
# 최근 처리 이력을 표로 확인합니다.

import re
import asyncio # 비동기 TTS 함수를 실행
import hashlib # 원본 음성의 SHA-256 해시값 계산
import json # 실행 기록을 JSONL 형식으로 저장
import os # DATABASE_URL 환경변수 조회
import pandas as pd
import tempfile # STT 입력을 위한 임시 WAV 파일 생성
import time # 처리 단계별 실행 시간 측정
import uuid # 요청별 고유 ID 생성
import wave # WAV 파일 헤더와 PCM 형식 검사
from datetime import datetime, timezone # UTC 시각 생성
from pathlib import Path # 파일 경로 처리
import numpy as np
import streamlit as st # 웹 기반 사용자 인터페이스
import sys
from dotenv import load_dotenv

load_dotenv() # .env 로딩

# 입력 파일 최대 크기를 10Mbytes로 제한
MAX_BYTES = 10 * 1024 * 1024
MIN_DURATION_SEC = 0.5 # 너무 짧은 녹음 기준 (초)
MAX_DURATION_SEC = 60 # 너무 긴 녹음 기준 (초, 처리시간 과다 방지)

# 실행이력을 저장할 jsonl파일 경로
LOG_PATH = Path(__file__).with_name('voice_runs_campus.jsonl')

SQL_DIR = Path(__file__).parent / 'sql' # SQL 파일 폴더 (실행 위치와 관계없이 이 파일 기준)

# 사용할 Whisper모델 크기
WHISPER_MODELS = ['tiny', 'base', 'small']

# Edge TTS에서 사용하는 한국어 여성 음성 모델 ID
TTS_VOICES = {
    '여성 (SunHi)': 'ko-KR-SunHiNeural',
    '남성 (InJoon)': 'ko-KR-InJoonNeural',
}

# 안내할 시설에 대한 답변 목록
FACILITIES = {
    "도서관": (
        "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다. "
        "1층에는 자료실과 대출·반납 데스크가 있고, 2층에는 열람실과 그룹 스터디룸이 있습니다. "
        "도서는 학생증으로 1인 최대 5권까지 14일간 대출할 수 있으며, 1회 연장이 가능합니다. "
    ),
    "학생식당": (
        "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다. "
        "한식, 양식, 분식 코너가 있으며 메뉴는 매주 월요일 학교 홈페이지에 공지됩니다. "
        "결제는 카드, 모바일 결제, 식권으로 가능하고 식권은 입구의 무인 발매기에서 구매할 수 있습니다. "
    ),
    "취업지원센터": (
        "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다. "
        "1:1 진로·취업 상담, 이력서와 자기소개서 첨삭, 모의면접 프로그램을 운영합니다. "
        "채용박람회, 기업 설명회, 취업 특강 일정은 센터 홈페이지와 게시판에 공지됩니다. "
    ),
    "행정실": (
        "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다. "
        "재학·졸업·성적 증명서 발급, 휴학과 복학 신청, 학적 변경, 등록금 관련 업무를 처리합니다. "
        "대부분의 증명서는 본관 로비의 무인 발급기나 학교 홈페이지에서 온라인으로도 발급할 수 있습니다. "
    ),
}

# 사용자가 줄여 말하는 경우 대응
ALIASES = {
    "도서관": ["도서관", "열람실", "스터디룸"],
    "학생식당": ["학생식당", "학식", "식당", "밥"],
    "취업지원센터": ["취업지원센터", "취업센터", "취업", "진로"],
    "행정실": ["행정실", "증명서", "휴학", "복학"],
}

STAGE_NAMES = {'stt': '음성 인식', 'reply': '응답 생성', 'tts': '음성 합성'}


# 별칭 키와 시설 키가 어긋나면 KeyError가 나므로 시작할 때 미리 확인
assert set(ALIASES) == set(FACILITIES), "ALIASES와 FACILITIES의 시설명이 다릅니다"

def normalize_text(text):
    if not text:
        return ''  # None 또는 빈 문자열은 빈 문자열로 통일

        # '학생 식당?' → '학생식당' 처럼 공백·문장부호 제거
    return re.sub(r'[\s.,!?~·]', '', text).lower()


# 인식된 문장에서 언급된 시설명을 등장 순서대로 찾는 함수
def find_facilities(transcript):
    clean = normalize_text(transcript)
    if not clean:
        return [] # 인식된 내용이 없으면 빈 목록 반환

    found = [] # (등장 위치, 시설명) 목록

    for name, aliases in ALIASES.items():
    # 별칭마다 문장 안의 위치를 찾고, 없는 경우(-1)는 제외
        positions = [clean.find(normalize_text(a)) for a in aliases]
        positions = [p for p in positions if p >= 0]

        if positions:
            found.append((min(positions), name)) # 가장 먼저 등장한 위치 기록

    found.sort() # 문장에서 먼저 언급된 시설이 앞에 오도록 정렬
    return [name for _, name in found]

# faster-whisper 음성 인식 모델을 CPU에 로딩하는 함수
@st.cache_resource(show_spinner='Whisper 모델을 로딩하고 있습니다...')
def load_stt_model(model_size):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device='cpu', compute_type='int8')

# WAV 데이터를 임시 파일로 저장한 후 실제 한국어 음성 인식 수행하는 함수
def transcribe_wav(audio, model_size):

    model = load_stt_model(model_size)

    # faster-whisper가 처리할 수 있도록 임시 WAV 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio) # 업로드된 원본 WAV 바이너리 기록
        wav_path = Path(f.name) # 생성된 임시 파일 경로

    try:

        # 전사 후 segment들과 부가정보 획득
        segments, info = model.transcribe(
            str(wav_path),
            language='ko', # 언어를 한국어로 인식
            vad_filter=True # 음성 활동 탐지로 무음 구간 처리
        )

        # 구간별 텍스트를 공백으로 연결하여 하나의 문장으로 생성
        transcript = ' '.join(segment.text.strip() for segment in segments).strip()

        return transcript

    finally:
        # 인식 성공/실패 여부와 관계없이 임시 WAV 파일 제거
        wav_path.unlink(missing_ok=True)

# 인식한 문장에 맞는 시설 안내 응답 문장을 생성하는 함수
def build_reply(transcript):

    # 무음이거나 인식 실패로 빈 문장인 경우
    if not normalize_text(transcript):
        return '음성 인식을 하지 못했습니다. 다시 말씀해 주세요.'
    facilities = find_facilities(transcript)

    # 안내 가능한 시설이 없으면 선택 가능한 시설 목록 안내
    if not facilities:
        names = ', '.join(FACILITIES)
        return f'죄송합니다, 해당 시설은 안내하지 않습니다. {names} 중 하나에 대해 질문해 주세요.'

    # 언급된 시설이 여러 개면 모두 이어서 안내
    return ' '.join(FACILITIES[name] for name in facilities)

# st.audio_input() 결과에서 WAV 바이너리를 꺼내는 함수
def read_audio_bytes(recorded):
    if recorded is None:
        return None # 녹음하지 않은 경우

    # Streamlit UploadFile 객체에서 전체 바이너리 획득
    return recorded.getvalue()

# 녹음 여부와 파일 용량을 검사하는 함수
def check_audio_size(audio):

    # 녹음 버튼을 누르지 않았거나 데이터가 비어 있는 경우
    if not audio:
        raise ValueError('먼저 마이크로 질문을 녹음해 주세요.')

    size_mb = len(audio) / (1024 * 1024) # 바이트 -> MB 환산

    if len(audio) > MAX_BYTES:
        raise ValueError(f'음성 파일이 너무 큽니다. ({size_mb:.1f} MB / 최대 10MB)')
    return round(size_mb, 2)


# 실제 WAV 파일인지 확인하고 음성 파일의 기본 정보를 추출하는 함수
# audio: 업로드되거나 녹음된 WAV파일의 바이너리 데이터 (bytes)
def validate_wav(audio):

    import io

    try:
        # 바이너리를 메모리 스트림으로 변환하여 WAV 헤더 읽기
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype() # 압축 형식 (NONE = 비압축 PCM)

    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('유효한 비압축 PCM WAV 파일을 입력해 주세요.') from e

    if compression != 'NONE' or channels not in (1, 2) or sample_rate <= 0 or frames <= 0:
        raise ValueError('모노/스테레오 비압축 PCM WAV 파일이 필요합니다.')

    return {
        'channels': channels,
        'sample_rate': sample_rate,
        'bit_depth': sample_width * 8,
        'duration_sec': round(frames / sample_rate, 2)
    }

# 녹음 길이가 처리 가능한 범위인지 검사하는 함수
def check_audio_duration(wav_info):

    duration = wav_info['duration_sec']

    # 버튼만 누르고 바로 멈춘 경우
    if duration < MIN_DURATION_SEC:
        raise ValueError(f'녹음이 너무 짧습니다. ({duration}초) 질문을 다시 녹음해 주세요.')

    # 너무 긴 녹음은 STT 처리시간이 크게 늘어나므로 제한
    if duration > MAX_DURATION_SEC:
        raise ValueError(f'녹음이 너무 깁니다. ({duration}초 / 최대 {MAX_DURATION_SEC}초)')

# 입력 검증 전체를 순서대로 수행하고 WAV 정보를 반환하는 함수
def validate_audio_input(audio):

    size_mb = check_audio_size(audio) # 1) 녹음 여부 및 용량
    wav_info = validate_wav(audio) # 2) WAV 형식
    check_audio_duration(wav_info) # 3) 녹음 길이

    wav_info['size_mb'] = size_mb # 이력 저장용으로 용량 정보 추가
    return wav_info

async def generate_tts_mp3(text, voice):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        mp3_path = Path(f.name)
    try:
        # 문장과 목소리ID를 전달한 후 MP3로 저장
        await edge_tts.Communicate(text=text, voice=voice).save(str(mp3_path))
        return mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)

# 비동기 Edge TTS 함수를 동기코드에서 실행하는 함수
def synthesize_speech(text, voice_id):
    mp3_bytes = asyncio.run(generate_tts_mp3(text, voice_id))

    if not mp3_bytes:
        raise RuntimeError('edge-tts에서 음성 데이터를 받지 못했습니다.')

    return mp3_bytes, 'audio/mpeg' # MP3 파일의 bytes와 MIME 변환

# STT > 응답 생성 > TTS를 순서대로 실행하고 단계별 시간(ms)과 오류를 기록하는 함수
def run_pipeline(audio, model_size, voice_id):
    # 결과 항목을 미리 모두 만들어 두어 오류가 나도 KeyError가 발생하지 않도록 함
    result = {
        'request_id': str(uuid.uuid4()),  # 요청을 구분할 고유 요청 ID
        'created_at': datetime.now(timezone.utc).isoformat(),  # 요청 시각 (UTC)
        'model_size': model_size,  # 사용한 Whisper 모델 크기
        'voice_id': voice_id,  # 사용한 TTS 목소리
        'transcript': '',  # 인식 문장 (화면 표시용, 이력에는 저장하지 않음)
        'reply': '',  # 응답 문장
        'output': None,  # 응답 음성 bytes
        'mime': None,  # 응답 음성 MIME
        'marks': {'stt_ms': None, 'reply_ms': None, 'tts_ms': None},  # 실행 안 된 단계는 None
        'total_ms': None,  # 전체 처리시간
        'error_stage': None,  # 오류가 난 단계 (성공 시 None)
        'error_type': None,  # 예외 클래스 이름
        'error_message': None,  # 화면 표시용 오류 내용
    }

    stage = None # 현재 실행 중인 단계 이름
    started = time.perf_counter() # 전체 처리 시작 시각

    try:

        # STT 단계 실행 후 소요 시간 측정
        stage  = 'stt'
        t0 = time.perf_counter()
        result['transcript'] = transcribe_wav(audio, model_size)
        result['marks']['stt_ms'] = round((time.perf_counter() - t0) * 1000, 2)

        # 규칙 기반 응답 생성 단계 실행 후 소요시간 측정
        stage = 'reply'
        t0 = time.perf_counter()
        result['reply'] = build_reply(result['transcript'])
        result['marks']['reply_ms'] = round((time.perf_counter() - t0) * 1000, 2)

        # 실제 TTS 음성 합성 단계 실행 후 소요 시간 측정
        stage = 'tts'
        t0 = time.perf_counter()
        result['output'], result['mime'] = synthesize_speech(result['reply'], voice_id)
        result['marks']['tts_ms'] = round((time.perf_counter() - t0) * 1000, 2)

    except Exception as e:
        # 오류가 나도 앱이 멈추지 않도록 실패 단계와 원인만 기록
        result['error_stage'] = stage
        result['error_type'] = type(e).__name__
        result['error_message'] = str(e)

    finally:
        # 성공/실패와 관계 없이 전체 처리 시간 기록 (최초 모델 로딩 시간 포함)
        result['total_ms'] = round((time.perf_counter() - started) * 1000, 2)

    return result

def show_result(result):

    # 오류가 발생한 단계와 원인 표시 (앱은 계속 동작)
    if result['error_stage']:
        stage_name = STAGE_NAMES.get(result['error_stage'], '알 수 없는 단계')
        st.error(f"{stage_name} 단계 오류: {result['error_type']}: {result['error_message']}")

    # 성공한 단계까지의 결과는 그대로 표시
    if result['transcript']:
        st.write('**인식된 질문:**', result['transcript'])

    if result['reply']:
        st.write('**안내 답변:**', result['reply'])

    if result['output']:
        st.audio(result['output'], format=result['mime'])

    # 구간별 소요시간 막대그래프 (실행되지 않은 단계는 0으로 표시)
    marks = result['marks']
    chart_data = pd.DataFrame({
        '구간': ['STT', '응답', 'TTS', '총 처리'],
        '소요시간(ms)': [marks['stt_ms'] or 0, marks['reply_ms'] or 0,
                     marks['tts_ms'] or 0, result['total_ms'] or 0],
    }).set_index('구간')

    st.bar_chart(chart_data)

# 파이프라인 결과에서 개인정보를 제외하고 저장할 항목만 골라 meta를 만드는 함수
def build_meta(result, audio, wav_info):

    marks = result['marks']

    meta = {
        # 실습 코드 DB 테이블과 같은 항목
        'request_id': result['request_id'],
        'created_at': result['created_at'],
        'status': 'error' if result['error_stage'] else 'success',
        'stt_ms': marks['stt_ms'] or 0.0, # DB 컬럼이 NOT NULL이므로 실행 안 된 단계는 0.0
        'reply_ms': marks['reply_ms'] or 0.0,
        'tts_ms': marks['tts_ms'] or 0.0,
        'total_ms': result['total_ms'] or 0.0,
        'model_version': f"whisper-{result['model_size']}/edge-tts/{result['voice_id']}",
        'audio_sha256': hashlib.sha256(audio).hexdigest(), # 음성 원본 대신 해시값만 보관

        # JSONL에만 추가로 남기는 항목 (DB 테이블 구조는 바꾸지 않음)
        'error_stage': result['error_stage'], # 실패한 단계 (0.0이 '실행 안 됨'인지 구분용)
        'error_type': result['error_type'], # 예외 클래스 이름만 기록
        'facilities': find_facilities(result['transcript']), # 질문 원문 대신 안내한 시설명만 기록
        'duration_sec': wav_info.get('duration_sec'), # 녹음 길이
        'size_mb': wav_info.get('size_mb'), # 녹음 용량
    }

    # 저장하지 않는 항목: transcript(질문 원문), output(음성), error_message(경로 등 노출 가능)
    return meta


# SQL 파일 내용을 문자열로 읽어오는 함수
def load_sql(file_name):
    return (SQL_DIR / file_name).read_text(encoding='utf-8')

# 실행 정보를 JSONL 파일에 추가하고 선택적으로 PostgreSQL에 저장하는 함수
def store_meta(meta):

    # JSONL은 JSON객체를 한 줄에 하나씩 기록하는 텍스트 파일
    try:
        with LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
    except OSError as e:
        st.warning(f'JSONL 실행 이력 저장 실패: {type(e).__name__}: {e}')

    # 환경변수의 DATABASE_URL 변수의 값 (없으면 DB 저장 생략)
    database_url = os.getenv('DATABASE_URL', '').strip()
    if not database_url:
        return

    try:

        import psycopg

        insert_sql = load_sql('insert_voice_run.sql') # INSERT 구문 읽기

        with psycopg.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                # 이름 기반 파라미터이므로 meta 딕셔너리를 그대로 전달
                # (meta에 JSONL 전용 항목이 더 있어도 SQL에 쓰인 키만 사용됨)
                cur.execute(insert_sql, meta)

    except Exception as e:
        # SQL 파일 없음, 테이블 없음, 접속 실패 등 모든 DB 오류는 경고만 표시
        st.warning(f'DB 실행 이력 저장 실패: {type(e).__name__}: {e}')

# ── 이력 조회 ────────────────────────────────────────

# JSONL 파일에서 최근 실행 이력을 최신순으로 읽어오는 함수
def load_recent_runs(limit=10):

    # 아직 한 번도 실행하지 않아 파일이 없는 경우
    if not LOG_PATH.exists():
        return []

    runs = [] # 읽어온 이력 목록

    with LOG_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # 빈 줄은 건너뜀
            try:
                runs.append(json.loads(line)) # JSON 문자열 → 딕셔너리
            except json.JSONDecodeError:
                continue # 저장 중 끊기는 등 깨진 줄은 건너뜀

    return runs[-limit:][::-1] # 마지막 limit개를 최신순으로 반환


# 최근 실행 이력을 표로 표시하는 함수
def show_recent_runs(limit=10):

    runs = load_recent_runs(limit)

    if not runs:
        st.info('아직 처리 이력이 없습니다.')
        return

    columns = ['created_at', 'status', 'error_stage', 'facilities',
               'stt_ms', 'reply_ms', 'tts_ms', 'total_ms', 'model_version', 'request_id']

    # reindex : 예전 형식의 이력에 없는 항목은 빈 값으로 채움 (KeyError 방지)
    df = pd.DataFrame(runs).reindex(columns=columns)

    # UTC로 저장된 시각을 한국 시간으로 변환해 표시
    df['created_at'] = (pd.to_datetime(df['created_at'], utc=True, errors='coerce')
                          .dt.tz_convert('Asia/Seoul')
                          .dt.strftime('%m-%d %H:%M:%S'))

    # 시설 목록을 쉼표 문자열로 표시
    df['facilities'] = df['facilities'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else '')

    df.columns = ['시각', '상태', '실패 단계', '안내 시설',
                  'STT(ms)', '응답(ms)', 'TTS(ms)', '총(ms)', '모델', '요청 ID']

    st.dataframe(df, hide_index=True, use_container_width=True)


# 전체 화면을 구성하고 처리 흐름을 연결하는 함수
def main():
    st.set_page_config(page_title='캠퍼스 시설 안내 음성 도우미', page_icon='🎙️', layout='centered')

    st.title('🎙️ 캠퍼스 시설 안내 음성 도우미')
    st.caption('마이크로 시설에 대해 질문하면 음성으로 안내합니다. '
               '실행 이력에는 질문 원문과 음성을 저장하지 않습니다.')

    # ── 사이드바 설정 ──
    with st.sidebar:
        st.header('설정')
        model_size = st.selectbox('Whisper 모델', WHISPER_MODELS, index=2,
                                  help='작을수록 빠르고, 클수록 정확합니다.')
        voice_name = st.selectbox('TTS 목소리', list(TTS_VOICES ))
        voice_id = TTS_VOICES [voice_name]

        st.divider()
        st.subheader('안내 가능한 시설')
        for name in FACILITIES:
            st.markdown(f'- {name}')
        st.caption('예) "도서관 몇 시까지 해요?", "학식 어디서 먹어요?"')

    # ── 음성 입력 ──
    st.subheader('1. 질문 녹음')
    recorded = st.audio_input('마이크 버튼을 눌러 질문을 녹음하세요')
    audio = read_audio_bytes(recorded)

    # ── 처리 실행 ──
    if st.button('질문 처리', type='primary', use_container_width=True):

        # ⑤ 입력 검증 (실패 시 경고만 표시하고 이후 단계 중단)
        try:
            wav_info = validate_audio_input(audio)
        except ValueError as e:
            st.warning(str(e))
            st.stop()

        # ⑥⑦ STT > 응답 > TTS 실행
        with st.spinner('질문을 처리하고 있습니다...'):
            result = run_pipeline(audio, model_size, voice_id)

        # ⑧ 이력 저장 (버튼을 눌렀을 때 한 번만)
        store_meta(build_meta(result, audio, wav_info))

        # 화면이 다시 그려져도 결과가 유지되도록 세션에 보관
        st.session_state['last_result'] = result
        st.session_state['last_wav_info'] = wav_info

    # ── 처리 결과 ──
    if 'last_result' in st.session_state:
        st.subheader('2. 처리 결과')

        with st.expander('입력 음성 정보'):
            st.json(st.session_state['last_wav_info'])

        show_result(st.session_state['last_result'])

    # ── 최근 이력 ──
    st.subheader('3. 최근 처리 이력')
    show_recent_runs()

if __name__ == '__main__':
    main()







































