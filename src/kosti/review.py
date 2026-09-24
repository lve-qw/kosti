"""Offline image review gallery for explicit, per-image expert annotations."""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .dicom import read_dicom
from .preparation import FIELDS, audit_records, read_json


def _thumbnail(image, max_side=420):
    pixels = np.clip(image.pixels * 255, 0, 255).astype(np.uint8)
    rendered = Image.fromarray(pixels, mode='L')
    rendered.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    result = io.BytesIO()
    rendered.save(result, format='PNG')
    return result.getvalue()


def build_review_gallery(audit_path, data_root, output):
    """Create a local HTML gallery and pixel previews; never export DICOM tags."""
    audit = read_json(audit_path)
    records = audit_records(audit)
    if audit.get('failed_count') or audit.get('failures'):
        raise ValueError('Resolve DICOM audit failures before review')
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.exists():
        raise ValueError('data_root must exist and review output must be new')
    groups = defaultdict(list)
    for record in records:
        groups[record['pixel_hash']].append(record)
    output.mkdir(parents=True)
    preview_dir = output / 'previews'
    preview_dir.mkdir()
    def prepare_group(work):
        index, copies = work
        preview = f'previews/{index:04d}.png'
        for position, copy in enumerate(copies):
            source = (root / copy['path']).resolve()
            if not source.is_relative_to(root):
                raise ValueError('Audit path escapes data_root')
            image = read_dicom(source)
            if any(getattr(image, key) != copy[key] for key in ('study_uid', 'image_uid', 'pixel_hash')):
                raise ValueError('DICOM identity changed since audit: ' + copy['path'])
            if position == 0:
                (output / preview).write_bytes(_thumbnail(image))
        return {'pixel_hash': copies[0]['pixel_hash'], 'preview': preview,
                'copies': [{key: row[key] for key in FIELDS[:4]} for row in copies],
                'study_count': len({r['study_uid'] for r in copies})}

    try:
        work = [(index, copies) for index, (_, copies) in enumerate(sorted(groups.items()))]
        with ThreadPoolExecutor(max_workers=8) as pool:
            items = list(pool.map(prepare_group, work))
        script_data = json.dumps({'fields': FIELDS, 'items': items}, ensure_ascii=False)
        # Escaping < prevents JSON content from prematurely ending the script tag.
        script_data = script_data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        page = _PAGE.replace('/* REVIEW_DATA */', script_data)
        (output / 'index.html').write_text(page, encoding='utf-8')
    except Exception:
        # A partial gallery can mislead review; remove only our newly created output.
        import shutil
        shutil.rmtree(output)
        raise
    return {'unique_images': len(items), 'files': len(records), 'html': str(output / 'index.html')}


_PAGE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка разметки DXA</title>
<style>
:root{font:16px/1.4 system-ui,sans-serif;color:#242b32;background:#f2f4f7}*{box-sizing:border-box}
body{margin:0}header{position:sticky;top:0;z-index:1;background:#fff;border-bottom:1px solid #d3d9df;padding:14px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}h1{font-size:19px;margin:0}button,.button{font:inherit;cursor:pointer;border:1px solid #82919d;border-radius:7px;background:#fff;padding:8px 12px}button.primary{background:#24577a;border-color:#24577a;color:#fff}button:disabled{opacity:.5;cursor:not-allowed}main{max-width:1120px;margin:20px auto;padding:0 16px}p{margin:8px 0}.muted{color:#576574}.card{background:#fff;border:1px solid #d3d9df;border-radius:12px;padding:20px;display:grid;grid-template-columns:minmax(250px,420px) 1fr;gap:22px}.image{background:#17212a;display:grid;place-items:center;min-height:300px}.image img{width:100%;max-height:560px;object-fit:contain}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}label{display:block;font-size:14px}input,select{display:block;width:100%;padding:9px;margin-top:4px;font:inherit;border:1px solid #8b9aa5;border-radius:6px}.checks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.checks label{border:1px solid #d5dce3;border-radius:7px;padding:8px}small{display:block;color:#576574}details{margin-top:14px;overflow-wrap:anywhere}footer{display:flex;gap:10px;margin-top:16px}#error{color:#a52c24;font-weight:600;min-height:1.5em}@media(max-width:760px){.card{grid-template-columns:1fr}.fields,.checks{grid-template-columns:1fr}}
</style></head>
<body><header><h1>Проверка снимков DXA</h1><span id="counter"></span><button id="save">Сохранить черновик</button><label class="button" for="restore">Загрузить черновик</label><input id="restore" type="file" accept="application/json" hidden><button id="pixel-export">Экспорт проверки пикселей</button><button id="export" class="primary">Экспорт разметки CSV</button></header>
<main><p class="muted">Просмотрите каждый снимок. Разметка переносится на точные копии пикселей, но остаётся отдельной строкой для каждого файла. Пустое значение означает неизвестную метку. Черновик сохраняйте перед закрытием страницы. Все действия локальны в браузере.</p><div id="error" role="alert"></div><article class="card"><div><div class="image"><img id="preview" alt="DXA снимок для проверки"></div><details><summary>Исходные файлы и исследования</summary><div id="copies"></div></details></div><div><p id="group"></p><div class="fields"><label>Область<select id="region"><option value="">Не выбрана</option><option value="spine">Поясничный отдел</option><option value="hip">Проксимальный отдел бедра</option></select></label><label>Сторона<select id="side"><option value="">Не выбрана / не применимо</option><option value="left">Левая</option><option value="right">Правая</option></select></label><label style="grid-column:1/-1">Источник меток и способ проверки<input id="source" placeholder="Например, разметка.xlsx, строка 3; эксперт подтвердил соответствие"></label></div><h3>Признаки нарушения</h3><div id="labels" class="checks"></div><label><input id="reviewed" type="checkbox" style="width:auto;display:inline"> Соответствие снимка и разметки проверено</label><label><input id="pixel-reviewed" type="checkbox" style="width:auto;display:inline"> В пикселях нет читаемых личных данных: проверено визуально</label><footer><button id="prev">← Предыдущий</button><button id="next">Следующий →</button></footer></div></article></main>
<script>const DATA=/* REVIEW_DATA */;
const NAMES={spine_position:'Позвоночник: укладка',spine_axis:'Позвоночник: ось',spine_artifact:'Позвоночник: предметы',hip_position:'Бедро: укладка',hip_roi:'Бедро: область интереса',spine_quality:'Позвоночник: итог',hip_quality:'Бедро: итог'};
const state=DATA.items.map(()=>({region:'',side:'',label_source:'',reviewed:false,pixel_reviewed:false,labels:Object.fromEntries(Object.keys(NAMES).map(k=>[k,'']))}));let index=0;
const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const item=DATA.items[index],s=state[index];$('counter').textContent=`${index+1} / ${DATA.items.length} · проверено ${state.filter(v=>v.reviewed).length}`;$('preview').src=item.preview;$('group').textContent=`Уникальное изображение ${index+1}; файлов-копий: ${item.copies.length}; исследований: ${item.study_count}`;$('copies').innerHTML=item.copies.map(c=>`<p>${esc(c.path)}<small>StudyUID: ${esc(c.study_uid)}</small></p>`).join('');$('region').value=s.region;$('side').value=s.side;$('source').value=s.label_source;$('reviewed').checked=s.reviewed;$('pixel-reviewed').checked=s.pixel_reviewed;$('labels').innerHTML=Object.entries(NAMES).map(([k,name])=>`<label>${esc(name)}<select data-label="${k}"><option value="">Неизвестно</option><option value="0">Нет нарушения</option><option value="1">Есть нарушение</option></select></label>`).join('');document.querySelectorAll('[data-label]').forEach(el=>{el.value=s.labels[el.dataset.label];el.disabled=!s.region||!el.dataset.label.startsWith(s.region+'_')});$('side').disabled=s.region!=='hip';$('prev').disabled=index===0;$('next').disabled=index===DATA.items.length-1}
$('region').onchange=e=>{const s=state[index];s.region=e.target.value;s.side='';for(const key of Object.keys(s.labels))if(!key.startsWith(s.region+'_'))s.labels[key]='';render()};$('side').onchange=e=>state[index].side=e.target.value;$('source').oninput=e=>state[index].label_source=e.target.value;$('reviewed').onchange=e=>{state[index].reviewed=e.target.checked;render()};$('pixel-reviewed').onchange=e=>{state[index].pixel_reviewed=e.target.checked;render()};$('labels').onchange=e=>{if(e.target.dataset.label)state[index].labels[e.target.dataset.label]=e.target.value};$('prev').onclick=()=>{index--;render()};$('next').onclick=()=>{index++;render()};
function download(name,type,content){const blob=new Blob([content],{type});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('save').onclick=()=>download('review-draft.json','application/json',JSON.stringify({schema_version:2,pixel_hashes:DATA.items.map(x=>x.pixel_hash),state}));
$('restore').onchange=async e=>{try{const raw=JSON.parse(await e.target.files[0].text());if(raw.schema_version!==2||JSON.stringify(raw.pixel_hashes)!==JSON.stringify(DATA.items.map(x=>x.pixel_hash))||!Array.isArray(raw.state)||raw.state.length!==state.length)throw Error('Черновик относится к другому аудиту');for(let i=0;i<state.length;i++){const s=raw.state[i];if(!['','spine','hip'].includes(s.region)||!['','left','right'].includes(s.side)||typeof s.label_source!=='string'||typeof s.reviewed!=='boolean'||typeof s.pixel_reviewed!=='boolean'||!s.labels||Object.keys(NAMES).some(k=>!['','0','1'].includes(s.labels[k])))throw Error('Некорректный черновик');state[i]=s}render();$('error').textContent='Черновик загружен'}catch(err){$('error').textContent=err.message}e.target.value=''};
function cell(v){const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}
$('pixel-export').onclick=()=>{const missing=state.flatMap((s,i)=>s.pixel_reviewed?[]:[i+1]);if(missing.length){$('error').textContent=`Проверьте текст на снимках: ${missing.slice(0,15).join(', ')}${missing.length>15?'…':''}`;return}$('error').textContent='';const rows=['pixel_hash,reviewed',...DATA.items.map(item=>`${cell(item.pixel_hash)},1`)];download('pixel-review.csv','text/csv;charset=utf-8','\ufeff'+rows.join('\r\n')+'\r\n')};$('export').onclick=()=>{const problems=[];state.forEach((s,i)=>{if(!s.reviewed||!s.label_source.trim()||!['spine','hip'].includes(s.region)||s.region==='hip'&&!['left','right'].includes(s.side)||s.region==='spine'&&s.side)problems.push(i+1);const parts=Object.keys(NAMES).filter(k=>k.startsWith(s.region+'_')&&!k.endsWith('_quality')).map(k=>s.labels[k]);const q=s.labels[s.region+'_quality'];if(q==='0'&&parts.includes('1')||q!==''&&parts.every(x=>x!=='')&&Number(q)!==Number(parts.includes('1')))problems.push(i+1)});if(problems.length){$('error').textContent=`Нужна проверка снимков: ${[...new Set(problems)].slice(0,15).join(', ')}${problems.length>15?'…':''}`;return}$('error').textContent='';const rows=[DATA.fields.join(',')];DATA.items.forEach((item,i)=>{const s=state[i];item.copies.forEach(c=>{const row={...c,region:s.region,side:s.side,label_source:s.label_source,reviewed:'1',...s.labels};rows.push(DATA.fields.map(k=>cell(row[k])).join(','))})});download('annotations.csv','text/csv;charset=utf-8','\ufeff'+rows.join('\r\n')+'\r\n')};render();
</script></body></html>'''
