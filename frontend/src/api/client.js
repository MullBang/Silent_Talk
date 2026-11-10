// client.js — axios 인스턴스 + API 호출 함수.
// 백엔드(FastAPI) baseURL: http://localhost:8000

import axios from 'axios';

import { baseMimeType, recordingFileName, validateBlob } from '../record/recordFlow.js';

/** 서비스 제한값 (설계문서 F-MVP-01). 클라이언트 1차 검증용. */
export const MAX_UPLOAD_SIZE_MB = 500;
export const MAX_DURATION_SEC = 180; // 3분
/** POST /api/record의 phrase_set으로 보낼 수 있는 지정 문장 묶음. 서버 기본은 'sentences'. */
export const PHRASE_SETS = Object.freeze(['sentences', 'ward']);
/** 내 입 모양 등록 이름 규칙 (서버와 같다). enroll_profile·enroll_command는 병실 묶음(ward)에만 보낸다. */
export const ENROLL_PROFILE_RE = /^[A-Za-z0-9_-]{1,32}$/;
/** enroll_command 형식 (기본 목록 W01~W20, 내 목록 U01~U20). 목록 안인지는 서버가 다시 확인한다. */
export const ENROLL_COMMAND_RE = /^[WU]\d{2}$/;
/** 내 목록 ID 규칙 (서버와 같다). command_list와 /api/ward/lists/{id} 주소에 쓴다. */
export const WARD_LIST_ID_RE = /^[A-Za-z0-9_-]{1,32}$/;
/** 내 목록 판(sha256, 소문자 16진수 64자). command_list_sha256과 PUT If-Match에 쓴다. */
export const WARD_LIST_SHA_RE = /^[0-9a-f]{64}$/;
/**
 * 서버가 '화면이 불러온 목록 판과 지금 목록이 다름'을 알릴 때의 문구 (POST /api/record 400,
 * backend/api/record.py MSG_LIST_CHANGED와 같다. backend/tests/test_ward_lists.py가 같은지 확인한다).
 */
export const LIST_CHANGED_MESSAGE = '목록이 바뀌었어요. 다시 불러와 주세요.';

/** 공용 axios 인스턴스. */
export const apiClient = axios.create({
  baseURL: import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 600000, // 대용량 업로드 대비 10분
});

/**
 * 클라이언트 1차 검증 오류(고정 한국어 문구)를 만든다. errorMessageFrom이 문구를 그대로 쓴다.
 * @param {string} message
 * @returns {Error & {isClientCheck: true}}
 */
function clientCheckError(message) {
  const err = new Error(message);
  err.isClientCheck = true;
  return err;
}

/**
 * File 객체의 재생 길이(초)를 메타데이터로 읽는다.
 * @param {File} file
 * @returns {Promise<number>} 재생 시간(초)
 */
function getVideoDurationSec(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(video.duration);
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('영상 메타데이터를 읽을 수 없습니다.'));
    };
    video.src = url;
  });
}

/**
 * 영상 파일을 업로드한다. (POST /api/upload)
 * 업로드 전 파일 크기/길이를 클라이언트에서 1차 검증한다.
 * @param {File} file
 * @param {(percent: number) => void} [onProgress] - 업로드 진행률(0~100) 콜백.
 * @returns {Promise<{ session_id: string, file_name: string, duration_sec: number }>}
 * @throws {Error} 크기 초과 / 길이 초과 시
 */
export async function uploadVideo(file, onProgress) {
  // ① 크기 검증 (500MB)
  const sizeMb = file.size / 1024 / 1024;
  if (sizeMb > MAX_UPLOAD_SIZE_MB) {
    throw clientCheckError(`${MAX_UPLOAD_SIZE_MB}MB 이하만 업로드 가능합니다.`);
  }

  // ② 길이 검증 (3분)
  let duration;
  try {
    duration = await getVideoDurationSec(file);
  } catch {
    throw clientCheckError('영상 파일을 읽을 수 없습니다. 지원 형식(MP4/AVI/MOV)을 확인해 주세요.');
  }
  if (duration > MAX_DURATION_SEC) {
    throw clientCheckError('3분 이하의 영상만 업로드 가능합니다.');
  }

  // ③ 업로드 (multipart/form-data)
  const form = new FormData();
  form.append('file', file);
  const { data } = await apiClient.post('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (typeof onProgress === 'function' && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return data;
}

/**
 * 녹화 요청의 등록 필드를 검사한다. 문제가 있으면 고정 한국어 문구, 없으면 null.
 * @param {string|undefined|null} phraseSet
 * @param {string|undefined|null} enrollProfile
 * @param {string|undefined|null} enrollCommand
 * @returns {string|null}
 */
export function enrollFieldProblem(phraseSet, enrollProfile, enrollCommand) {
  const hasProfile = enrollProfile !== undefined && enrollProfile !== null && enrollProfile !== '';
  const hasCommand = enrollCommand !== undefined && enrollCommand !== null && enrollCommand !== '';
  if (!hasProfile && !hasCommand) return null;
  if (phraseSet !== 'ward') return '입 모양 등록은 병실 명령 녹화에서만 쓸 수 있습니다.';
  if (!hasProfile) return '등록할 명령을 보내려면 등록 이름이 필요합니다.';
  if (typeof enrollProfile !== 'string' || !ENROLL_PROFILE_RE.test(enrollProfile)) {
    return '등록 이름은 영문·숫자·_·- 1~32자로 적어 주세요.';
  }
  if (hasCommand && (typeof enrollCommand !== 'string' || !ENROLL_COMMAND_RE.test(enrollCommand))) {
    return '등록할 명령 ID가 올바르지 않습니다.';
  }
  return null;
}

/**
 * 녹화 요청의 내 목록 필드(command_list)를 검사한다. 문제가 있으면 고정 한국어 문구, 없으면 null.
 * 내 목록은 병실 묶음(ward)에만 보낼 수 있고, 내 목록 명령(U..)은 내 목록과 함께, 기본 목록 명령(W..)은 내 목록 없이 보낸다.
 * @param {string|undefined|null} phraseSet
 * @param {string|undefined|null} commandList
 * @param {string|undefined|null} enrollCommand
 * @returns {string|null}
 */
export function commandListProblem(phraseSet, commandList, enrollCommand, commandListSha256) {
  const hasSha = commandListSha256 !== undefined && commandListSha256 !== null && commandListSha256 !== '';
  const hasList = commandList !== undefined && commandList !== null && commandList !== '';
  const command = typeof enrollCommand === 'string' ? enrollCommand : '';
  if (!hasList) {
    if (hasSha) return '목록 지문은 내 목록과 함께 보내야 합니다.';
    return command.startsWith('U') ? '내 목록 명령은 내 목록과 함께 보내야 합니다.' : null;
  }
  if (phraseSet !== 'ward') return '내 목록은 병실 명령 녹화에서만 쓸 수 있습니다.';
  if (typeof commandList !== 'string' || !WARD_LIST_ID_RE.test(commandList)) return '내 목록 이름이 올바르지 않습니다.';
  if (command.startsWith('W')) return '기본 목록 명령은 내 목록과 함께 보낼 수 없습니다.';
  if (hasSha && (typeof commandListSha256 !== 'string' || !WARD_LIST_SHA_RE.test(commandListSha256))) {
    return '목록 지문 형식이 올바르지 않습니다.';
  }
  return null;
}

/**
 * 서버 오류가 '목록이 바뀌었어요'(화면이 불러온 목록 판과 서버의 지금 목록이 다름, 400)인지.
 * @param {any} err axios 오류.
 * @returns {boolean}
 */
export function isListChangedError(err) {
  try {
    return err?.response?.status === 400 && err?.response?.data?.detail === LIST_CHANGED_MESSAGE;
  } catch {
    return false;
  }
}

/**
 * 브라우저 녹화 영상을 올리고 분석 Job을 시작한다. (POST /api/record, 6단계)
 * 서버가 실제 길이를 재서 30초 미만만 접수하고 즉시 결과 대신 {job_id, session_id, status:'processing',
 * duration_sec}를 돌려준다. 결과는 getResult(session_id) 폴링으로 받는다.
 * 파일 파트의 MIME은 코덱 매개변수를 뗀 기본형(video/webm|video/mp4), 파일명은 recording.webm|.mp4다.
 * options.phraseSet을 주면 phrase_set 필드로 지정 문장 묶음을 고른다(병실 화면은 'ward').
 * 주지 않으면 필드를 보내지 않아 서버 기본(sentences) 요청과 똑같다.
 * options.enrollProfile(내 입 모양 등록 이름)·enrollCommand(등록할 명령 ID)는 phraseSet 'ward'일 때만 보낼 수 있고,
 * 비어 있으면 필드를 보내지 않는다. enrollCommand가 있으면 등록(저장), 없으면 등록과 비교(match)다.
 * options.commandList(내 목록 ID)도 phraseSet 'ward'일 때만 보내고, 비어 있으면 보내지 않는다(기본 목록 요청은 오늘과 같다).
 * options.commandListSha256(불러온 목록 판)은 commandList와 함께만 보낸다. 서버의 지금 목록과 다르면 서버가 400
 * (LIST_CHANGED_MESSAGE)으로 거절한다(등록 화면이 다른 탭에서 고친 목록에 옛 입 모양을 저장하지 않게).
 * @param {Blob} blob 녹화 Blob.
 * @param {string} mime MediaRecorder MIME (예: 'video/webm;codecs=vp9').
 * @param {(percent: number) => void} [onProgress] 업로드 진행률(0~100) 콜백.
 * @param {{phraseSet?: 'sentences'|'ward', enrollProfile?: string, enrollCommand?: string, commandList?: string,
 *   commandListSha256?: string}} [options]
 * @returns {Promise<{ job_id: string, session_id: string, status: string, duration_sec: number }>}
 * @throws {Error} 빈 Blob·모르는 묶음(isClientCheck=true) 또는 axios 오류(서버 400/503 detail 포함).
 */
export async function uploadRecording(blob, mime, onProgress, options = {}) {
  const check = validateBlob(blob);
  if (!check.ok) {
    const err = new Error(check.message);
    err.isClientCheck = true;
    throw err;
  }
  const phraseSet = options?.phraseSet;
  if (phraseSet !== undefined && phraseSet !== null && !PHRASE_SETS.includes(phraseSet)) {
    const err = new Error('지원하지 않는 문장 묶음입니다.');
    err.isClientCheck = true;
    throw err;
  }
  const enrollProfile = options?.enrollProfile;
  const enrollCommand = options?.enrollCommand;
  const enrollProblem = enrollFieldProblem(phraseSet, enrollProfile, enrollCommand);
  if (enrollProblem) throw clientCheckError(enrollProblem);
  const commandList = options?.commandList;
  const commandListSha256 = options?.commandListSha256;
  const listProblem = commandListProblem(phraseSet, commandList, enrollCommand, commandListSha256);
  if (listProblem) throw clientCheckError(listProblem);
  const type = baseMimeType(mime || blob.type);
  const part = blob.type === type ? blob : new Blob([blob], { type });
  const form = new FormData();
  form.append('file', part, recordingFileName(type));
  if (phraseSet !== undefined && phraseSet !== null) form.append('phrase_set', phraseSet);
  if (commandList) form.append('command_list', commandList);
  if (commandList && commandListSha256) form.append('command_list_sha256', commandListSha256);
  if (enrollProfile) form.append('enroll_profile', enrollProfile);
  if (enrollCommand) form.append('enroll_command', enrollCommand);
  const { data } = await apiClient.post('/api/record', form, {
    timeout: 120000,
    onUploadProgress: (e) => {
      if (typeof onProgress === 'function' && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return data;
}

/**
 * 추론 Job을 시작한다. (POST /api/infer)
 * 즉시 결과를 반환하지 않고 {job_id, session_id, status:'processing'}을 받는다.
 * @param {string} sessionId
 * @returns {Promise<{ job_id: string, session_id: string, status: string }>}
 */
export async function startInfer(sessionId) {
  const { data } = await apiClient.post('/api/infer', { session_id: sessionId });
  return data;
}

/**
 * 추론 결과를 조회한다. (GET /api/result/{sessionId})
 * @param {string} sessionId
 * @returns {Promise<{ session_id: string, status: string, progress: number,
 *   error_message: string|null, results: Array|null }>}
 */
export async function getResult(sessionId) {
  const { data } = await apiClient.get(`/api/result/${sessionId}`);
  return data;
}

/**
 * 등록 이름을 주소에 넣기 전에 검사한다 (경로 이동 등 이상한 값은 보내지 않는다).
 * @param {string} profile
 * @returns {string} 주소용으로 인코딩한 이름.
 * @throws {Error} 규칙에 맞지 않으면 isClientCheck 오류.
 */
function profilePath(profile) {
  if (typeof profile !== 'string' || !ENROLL_PROFILE_RE.test(profile)) {
    throw clientCheckError('등록 이름은 영문·숫자·_·- 1~32자로 적어 주세요.');
  }
  return `/api/ward/profiles/${encodeURIComponent(profile)}`;
}

/**
 * 등록 이름공간 주소 뒤에 붙일 ?list=<내 목록 ID>. 기본 목록(''·undefined)이면 ''(예전 주소 그대로).
 * @param {string|undefined|null} list
 * @returns {string}
 * @throws {Error} 규칙에 맞지 않는 목록 ID면 isClientCheck 오류.
 */
function listQuery(list) {
  if (list === undefined || list === null || list === '') return '';
  if (typeof list !== 'string' || !WARD_LIST_ID_RE.test(list)) throw clientCheckError('내 목록 이름이 올바르지 않습니다.');
  return `?list=${encodeURIComponent(list)}`;
}

/**
 * 내 입 모양 등록 목록. (GET /api/ward/profiles[?list=ID])
 * @param {string} [list] 내 목록 ID (없으면 기본 목록의 등록).
 * @returns {Promise<{profiles: Array<{profile:string, count:number, complete:boolean, status:string, updated:string|null}>}>}
 */
export async function getWardProfiles(list) {
  const { data } = await apiClient.get(`/api/ward/profiles${listQuery(list)}`, { timeout: 15000 });
  return data;
}

/**
 * 등록 하나의 상태. (GET /api/ward/profiles/{profile}[?list=ID]) 서버에 없는 이름(404)은 status 'missing'인 빈 등록으로 돌려준다.
 * @param {string} profile
 * @param {string} [list] 내 목록 ID (없으면 기본 목록의 등록).
 * @returns {Promise<{profile:string, count:number, complete:boolean, status:string, templates:Record<string, object>}>}
 */
export async function getWardProfile(profile, list) {
  const path = profilePath(profile) + listQuery(list);
  try {
    const { data } = await apiClient.get(path, { timeout: 15000 });
    return data;
  } catch (err) {
    if (err?.response?.status === 404) return { profile, count: 0, complete: false, status: 'missing', templates: {} };
    throw err;
  }
}

/**
 * 등록을 지운다. (DELETE /api/ward/profiles/{profile}[?list=ID])
 * @param {string} profile
 * @param {string} [list] 내 목록 ID (없으면 기본 목록의 등록).
 * @returns {Promise<{deleted: boolean}>}
 */
export async function deleteWardProfile(profile, list) {
  const path = profilePath(profile) + listQuery(list);
  const { data } = await apiClient.delete(path, { timeout: 15000 });
  return data;
}

/**
 * 목록 하나(기본 목록 '' 또는 내 목록 ID)에 묶인 등록 API. 화면은 지금 쓰는 목록으로 한 번 만들어 쓴다.
 * 목록 ID가 규칙에 맞지 않으면 각 호출이 isClientCheck 오류로 거절된다(서버로 보내지 않음).
 * @param {string} [list] ''이면 기본 목록.
 * @returns {{list:string, getWardProfiles: () => Promise<any>, getWardProfile: (profile:string) => Promise<any>,
 *   deleteWardProfile: (profile:string) => Promise<any>}}
 */
export function wardProfileApi(list = '') {
  const id = typeof list === 'string' ? list : '';
  return Object.freeze({
    list: id,
    getWardProfiles: () => getWardProfiles(id || undefined),
    getWardProfile: (profile) => getWardProfile(profile, id || undefined),
    deleteWardProfile: (profile) => deleteWardProfile(profile, id || undefined),
  });
}

/**
 * 내 목록 ID를 주소에 넣기 전에 검사한다.
 * @param {string} id
 * @returns {string} /api/ward/lists/{id}
 * @throws {Error} 규칙에 맞지 않으면 isClientCheck 오류.
 */
function listPath(id) {
  if (typeof id !== 'string' || !WARD_LIST_ID_RE.test(id)) {
    throw clientCheckError('목록 ID는 영문·숫자·_·- 1~32자로 적어 주세요.');
  }
  return `/api/ward/lists/${encodeURIComponent(id)}`;
}

/**
 * 내 목록들. (GET /api/ward/lists)
 * @returns {Promise<{lists: Array<{id:string, title:string, count:number, emergency_count:number, updated:string, sha256:string}>}>}
 */
export async function getWardLists() {
  const { data } = await apiClient.get('/api/ward/lists', { timeout: 15000 });
  return data;
}

/**
 * 내 목록 하나와 그 검사 결과. (GET /api/ward/lists/{id}) 없으면 404 오류를 그대로 던진다.
 * @param {string} id
 * @returns {Promise<{schema:string, id:string, title:string, created:string, updated:string,
 *   commands:Array<{id:string, text:string, display:string, category:string, emergency:boolean, cancel:boolean}>, sha256:string,
 *   checks:{errors:Array, warnings:Array}}>}
 */
export async function getWardList(id) {
  const path = listPath(id);
  const { data } = await apiClient.get(path, { timeout: 15000 });
  return data;
}

/**
 * PUT 저장 조건 헤더. create면 If-None-Match: *(없을 때만 만듦), ifMatch(sha256)면 If-Match(그 판일 때만 고침).
 * 둘 다 없으면 {}(조건 없이 저장). 맞지 않으면 서버가 409로 거절한다.
 * @param {{create?: boolean, ifMatch?: string|null}} [cond]
 * @returns {Record<string, string>}
 * @throws {Error} 둘을 함께 주거나 sha256 형식이 틀리면 isClientCheck 오류.
 */
export function listSaveHeaders(cond = {}) {
  const create = cond?.create === true;
  const ifMatch = typeof cond?.ifMatch === 'string' && cond.ifMatch ? cond.ifMatch : null;
  if (create && ifMatch) throw clientCheckError('새 목록 저장과 고치기 조건을 함께 쓸 수 없습니다.');
  if (create) return { 'If-None-Match': '*' };
  if (ifMatch) {
    if (!WARD_LIST_SHA_RE.test(ifMatch)) throw clientCheckError('목록 지문 형식이 올바르지 않습니다.');
    return { 'If-Match': `"${ifMatch}"` };
  }
  return {};
}

/**
 * 내 목록을 저장한다. (PUT /api/ward/lists/{id}) 오류가 있으면 서버가 400 {detail, errors, warnings}로 거절한다.
 * cond.create(새 목록)·cond.ifMatch(불러온 판의 sha256)를 주면 조건 헤더를 보낸다: 이미 있는 ID로 새 목록을 만들거나
 * 다른 곳에서 바뀐 목록을 덮어쓰려 하면 서버가 409로 거절한다.
 * @param {string} id
 * @param {{title:string, commands:Array<{text:string, display:string, category:string, emergency:boolean, cancel:boolean}>}} body
 * @param {{create?: boolean, ifMatch?: string|null}} [cond]
 * @returns {Promise<object>} 저장한 목록 + warnings.
 */
export async function saveWardList(id, body, cond = {}) {
  const path = listPath(id);
  const headers = listSaveHeaders(cond);
  const { data } = await apiClient.put(path, body, { timeout: 15000, headers });
  return data;
}

/**
 * 내 목록을 지운다 (그 목록의 입 모양 등록도 서버가 함께 지운다). (DELETE /api/ward/lists/{id})
 * @param {string} id
 * @returns {Promise<{deleted: boolean}>}
 */
export async function deleteWardList(id) {
  const path = listPath(id);
  const { data } = await apiClient.delete(path, { timeout: 15000 });
  return data;
}

/**
 * 저장하지 않고 검사만 한다. (POST /api/ward/lists/check)
 * @param {{title:string, commands:Array<object>}} body
 * @returns {Promise<{errors:Array, warnings:Array, normalized:object}>}
 */
export async function checkWardList(body) {
  const { data } = await apiClient.post('/api/ward/lists/check', body, { timeout: 15000 });
  return data;
}
