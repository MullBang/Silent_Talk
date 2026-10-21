"""Bind actual recorded visual observations to the exact120 evidence rows."""
from datetime import datetime, timezone
from pathlib import Path
import json
from build_review import BASE, bind, read, require, sha


def main():
    media_path = BASE / 'results/media_manifest.json'
    notes_path = BASE / 'visual_observations.json'
    media, notes = read(media_path), read(notes_path)
    observed = {Path(row['path']).resolve(): row for row in notes['observations']}
    require(len(observed) == notes['source_count'] == 30, 'Need30 actually viewed contact sheets')
    sheets = {Path(clip['source_contact_sheet']['path']).resolve() for clip in media['clips']}
    require(set(observed) == sheets, 'Viewed source coverage differs')
    rows = []
    for clip in media['clips']:
        sheet = clip['source_contact_sheet']
        require(sha(sheet['path']) == sheet['sha256'], 'Contact sheet changed')
        observation = observed[Path(sheet['path']).resolve()]
        rows.append({'id': clip['id'], 'source_stem': clip['source_stem'], 'split': clip['split'],
                     'finding': observation['finding'], 'note': observation['note'],
                     'provisional_finding': observation.get('provisional_finding'),
                     'roi_images': [sample['image'] for sample in clip['roi_samples']],
                     'face_context': clip['face_context']['image'], 'contact_sheet': sheet,
                     'review_method': 'Actual model visual inspection of first/middle/last PNG samples in contact sheet',
                     'human_listening_confirmed': False, 'training_ready': False})
    result = {'schema': 'lipnet-step06-visual-review-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'media_manifest': bind(media_path), 'observations': bind(notes_path), 'code': bind(Path(__file__)),
              'diagnostic_followup': bind(BASE / 'diagnosis/lip_J_6_M_03_C646_A_004/result.json'),
              'records': rows, 'counts': {'sources': 30, 'sentences': 120, 'roi_samples_viewed': 360,
                                       'context_images_viewed': 120, 'mouth_visible_in_reviewed_samples': 120,
                                       'sentences_with_strong_texture_observation': 32},
              'human_listening_confirmed_count': 0, 'training_ready': False,
              'limits': ['Model visual inspection is not a human-listening or lip-sync certificate.',
                         'Three sampled frames do not establish full-sequence visual quality.',
                         'Strong texture is an observation, not an established corrupted video or label error.',
                         'The extra native-frame diagnosis rereads one affected frame with the same decoder; camera/codec/decoder causes remain unresolved.']}
    output = BASE / 'results/visual_review.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
