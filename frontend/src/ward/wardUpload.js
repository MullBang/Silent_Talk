// wardUpload.js — 병실 화면(#ward)과 내 입 모양 등록 화면(#enroll)이 함께 쓰는 "녹화 → 올리기 → 결과 기다리기" (node --test로 검사).
// 녹화를 1차 검사하고 POST /api/record(phrase_set='ward')로 올린 뒤 약 300ms 간격으로 결과를 폴링한다.
// 등록 이름이 있으면 enroll_profile을, 등록할 명령이 있으면 enroll_command도 함께 보낸다.
// 내 목록을 쓰면 command_list(목록 ID)도 보낸다. 기본 목록이면 보내지 않아 요청이 예전과 같다.
// 등록 화면은 불러온 목록 판(command_list_sha256)도 보내, 그사이 목록이 바뀌었으면 서버가 저장하지 않고 거절한다.

import { getResult, isListChangedError, uploadRecording } from '../api/client.js';
import { recordUploadErrorMessage, validateRecording } from '../record/recordFlow.js';
import { WARD_PHRASE_SET } from './wardCommands.js';
import { pollWardResult } from './wardPoll.js';

/**
 * POST /api/record의 병실 묶음 옵션. 빈 등록 이름·명령·목록은 넣지 않는다(서버 요청이 오늘과 같다).
 * 목록 판(commandListSha256)은 내 목록과 함께일 때만 넣는다.
 * @param {{enrollProfile?: string|null, enrollCommand?: string|null, commandList?: string|null,
 *   commandListSha256?: string|null}} [opts]
 * @returns {{phraseSet: 'ward', commandList?: string, commandListSha256?: string, enrollProfile?: string, enrollCommand?: string}}
 */
export function wardRecordOptions({ enrollProfile = '', enrollCommand = '', commandList = '', commandListSha256 = '' } = {}) {
  const options = { phraseSet: WARD_PHRASE_SET };
  if (typeof commandList === 'string' && commandList) options.commandList = commandList;
  if (options.commandList && typeof commandListSha256 === 'string' && commandListSha256) {
    options.commandListSha256 = commandListSha256;
  }
  if (typeof enrollProfile === 'string' && enrollProfile) options.enrollProfile = enrollProfile;
  if (options.enrollProfile && typeof enrollCommand === 'string' && enrollCommand) options.enrollCommand = enrollCommand;
  return options;
}

/**
 * 녹화 하나를 올리고 결과를 기다린다.
 * @param {object} opts
 * @param {Blob} opts.blob 녹화 Blob.
 * @param {string} opts.mime MediaRecorder MIME.
 * @param {number} opts.durationMs 녹화기가 잰 길이(ms).
 * @param {string} [opts.enrollProfile] 내 입 모양 등록 이름(''이면 보내지 않음).
 * @param {string} [opts.enrollCommand] 등록할 명령 ID(등록 화면만).
 * @param {string} [opts.commandList] 내 목록 ID(''이면 기본 목록, 보내지 않음).
 * @param {string} [opts.commandListSha256] 불러온 내 목록 판(등록 화면). 서버의 지금 목록과 다르면 거절되고
 *   결과가 {kind:'error', listChanged:true}다.
 * @param {() => boolean} [opts.isCancelled] true면 바로 멈춘다(화면 이탈·새 녹화).
 * @param {{upload?: Function, fetchResult?: Function, poll?: Function}} [opts.deps] 테스트용 주입.
 * @returns {Promise<{kind:'invalid', message:string}|{kind:'done', results:Array, warnings:string[]}
 *   |{kind:'error', message:string, listChanged?: true}|{kind:'timeout'}|{kind:'cancelled'}>}
 */
export async function analyzeWardRecording({
  blob, mime, durationMs, enrollProfile = '', enrollCommand = '', commandList = '', commandListSha256 = '',
  isCancelled = () => false, deps = {},
}) {
  const upload = deps.upload || uploadRecording;
  const fetchResult = deps.fetchResult || getResult;
  const poll = deps.poll || pollWardResult;
  const check = validateRecording(blob, durationMs);
  if (!check.ok) return { kind: 'invalid', message: check.message };
  try {
    const data = await upload(blob, mime, undefined, wardRecordOptions({
      enrollProfile, enrollCommand, commandList, commandListSha256,
    }));
    if (isCancelled()) return { kind: 'cancelled' };
    if (!data?.session_id) throw new Error('missing session_id');
    const outcome = await poll({ fetchResult: () => fetchResult(data.session_id), isCancelled });
    if (outcome?.kind === 'done') return { kind: 'done', results: outcome.results, warnings: outcome.warnings || [] };
    if (outcome?.kind === 'error') return { kind: 'error', message: outcome.message };
    if (outcome?.kind === 'timeout') return { kind: 'timeout' };
    return { kind: 'cancelled' };
  } catch (err) {
    if (isCancelled()) return { kind: 'cancelled' };
    const out = { kind: 'error', message: recordUploadErrorMessage(err) };
    return isListChangedError(err) ? { ...out, listChanged: true } : out;
  }
}
