(() => {
  'use strict';
  const byId = (id) => document.getElementById(id);
  const state = { ready: false, busy: false, files: [], result: null, selected: 0 };
  const status = byId('model-status');
  const notice = byId('notice');
  const runButton = byId('run-button');
  const input = byId('input-files');
  const dropzone = byId('dropzone');

  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = String(content);
    return element;
  }
  function setNotice(message, error = false) {
    notice.textContent = message;
    notice.classList.toggle('error', error);
  }
  function updateUpload() {
    const count = state.files.length;
    byId('file-summary').textContent = count ? `${count} файл(ов): ${state.files.slice(0, 2).map(file => file.name).join(', ')}${count > 2 ? '…' : ''}` : 'Файлы не выбраны';
    runButton.disabled = !state.ready || !count || state.busy;
    runButton.textContent = state.busy ? 'Обрабатываем…' : 'Проверить снимки ↗';
  }
  async function checkReady() {
    status.textContent = 'Проверяем модель…';
    status.className = 'status-pill status-loading';
    try {
      const response = await fetch('/ready', { cache: 'no-store' });
      state.ready = response.ok;
      status.textContent = response.ok ? 'Модель подключена' : 'Модель качества не подключена';
      status.className = `status-pill ${response.ok ? 'status-ready' : 'status-error'}`;
      if (!response.ok) setNotice('Обработка будет доступна после подключения проверенной модели качества.');
      else if (!state.result) setNotice('Можно загрузить DICOM или ZIP.');
    } catch {
      state.ready = false;
      status.textContent = 'Сервер недоступен';
      status.className = 'status-pill status-error';
      setNotice('Не удалось связаться с локальным сервером.', true);
    }
    updateUpload();
  }
  function fileLabel(path) { return String(path).split('/').at(-1) || String(path); }
  function diagnosticsFor(row) {
    return state.result.diagnostics.filter(item => item.path_to_study === row.path_to_study);
  }
  function addData(list, label, value) {
    list.append(node('dt', '', label), node('dd', '', value || '—'));
  }
  function renderDetail() {
    const detail = byId('detail');
    detail.replaceChildren();
    const row = state.result?.rows[state.selected];
    if (!row) return;
    detail.append(node('p', 'eyebrow', `Снимок ${state.selected + 1} / ${state.result.rows.length}`));
    detail.append(node('h3', '', fileLabel(row.path_to_study)));
    detail.append(node('p', 'path', row.path_to_study));
    const image = node('div', 'preview');
    const preview = state.result.previews[String(state.selected)];
    if (preview) {
      const picture = node('img');
      picture.src = preview;
      picture.alt = `Снимок ${fileLabel(row.path_to_study)}`;
      image.append(picture);
    } else image.append(node('p', 'no-preview', 'Изображение недоступно'));
    detail.append(image);
    const failed = row.processing_status !== 'Success';
    const card = node('section', `quality-card${failed || row.quality_class === 1 ? ' problem' : ''}`);
    card.append(node('p', 'quality-title', failed ? 'Не обработано' : row.quality_class === 1 ? 'Обнаружено нарушение' : 'Нарушения не обнаружены'));
    card.append(node('p', '', failed ? 'Для этого файла нет результата модели.' : row.anatomical_region));
    detail.append(card);
    if (!failed) {
      const list = node('dl', 'data-list');
      addData(list, 'Область', row.anatomical_region);
      addData(list, 'Класс качества', row.quality_class === 1 ? '1 — есть нарушение' : '0 — нарушений не выявлено');
      addData(list, 'Вероятность общего нарушения', row.quality_prob === null ? '—' : `${(Number(row.quality_prob) * 100).toFixed(1)} %`);
      addData(list, 'Время обработки', `${Number(row.time_of_processing).toFixed(2)} с`);
      detail.append(list);
      detail.append(node('h3', '', 'Типы нарушений'));
      if (row.violation_type) {
        const types = node('ul', 'violation-list');
        row.violation_type.split('; ').forEach(label => types.append(node('li', '', label)));
        detail.append(types);
      } else detail.append(node('p', 'subdued', 'Пороговые нарушения не выявлены.'));
      detail.append(node('p', 'detail-footnote', 'Класс определяется порогами типов нарушений. Вероятность общего нарушения — отдельный выход модели; эти оценки могут различаться.'));
    }
    const metadata = node('dl', 'data-list');
    addData(metadata, 'UID исследования', row.study_uid);
    addData(metadata, 'UID изображения', row.image_uid);
    detail.append(metadata);
    const messages = diagnosticsFor(row);
    if (messages.length) {
      const box = node('section', 'diagnostics');
      box.append(node('h3', '', 'Диагностика'));
      messages.forEach(item => {
        if (item.error) box.append(node('p', 'error-text', `${item.error_type || 'Ошибка'}: ${item.error}`));
        (item.warnings || []).forEach(warning => box.append(node('p', '', warning)));
      });
      detail.append(box);
    }
  }
  function renderRows() {
    const list = byId('rows');
    list.replaceChildren();
    const search = byId('search').value.trim().toLowerCase();
    let visible = 0;
    state.result.rows.forEach((row, index) => {
      if (search && ![row.path_to_study, row.study_uid, row.image_uid].some(value => String(value || '').toLowerCase().includes(search))) return;
      visible += 1;
      const button = node('button', `row-button${index === state.selected ? ' selected' : ''}`);
      button.type = 'button';
      button.setAttribute('aria-label', `Открыть результат ${fileLabel(row.path_to_study)}`);
      const label = node('span');
      label.append(node('span', 'row-title', fileLabel(row.path_to_study)), node('span', 'row-subtitle', row.anatomical_region || row.path_to_study));
      const failed = row.processing_status !== 'Success';
      const badge = node('span', `row-state${failed || row.quality_class === 1 ? ' problem' : ''}`, failed ? 'Ошибка' : row.quality_class === 1 ? 'Нарушение' : 'Без нарушений');
      button.append(label, badge);
      button.addEventListener('click', () => { state.selected = index; renderRows(); renderDetail(); });
      const item = node('div');
      item.setAttribute('role', 'listitem');
      item.append(button);
      list.append(item);
    });
    if (!visible) list.append(node('p', 'empty-list', 'По запросу ничего не найдено.'));
  }
  function renderResults() {
    const rows = state.result.rows;
    const errors = rows.filter(row => row.processing_status !== 'Success').length;
    byId('results-count').textContent = `${rows.length} снимков · ${rows.length - errors} обработано · ${errors} с ошибкой`;
    byId('results').classList.remove('hidden');
    state.selected = 0;
    byId('search').value = '';
    renderRows();
    renderDetail();
  }
  async function run(files) {
    if (!state.ready || !files.length || state.busy) return;
    state.busy = true;
    updateUpload();
    setNotice('Загружаем и проверяем снимки. Это может занять несколько минут.');
    try {
      const form = new FormData();
      files.forEach(file => form.append('files', file, file.name));
      const response = await fetch('/v1/review', { method: 'POST', body: form, cache: 'no-store' });
      if (!response.ok) {
        let reason = `Сервер вернул ошибку ${response.status}`;
        try { const body = await response.json(); if (typeof body.detail === 'string') reason = body.detail; } catch { /* Keep HTTP status. */ }
        throw new Error(reason);
      }
      const result = await response.json();
      if (!Array.isArray(result.rows) || !Array.isArray(result.diagnostics) || typeof result.csv !== 'string' || !result.previews) throw new Error('Некорректный ответ сервера');
      state.result = result;
      renderResults();
      setNotice('Обработка завершена. Выберите снимок для просмотра.');
      byId('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
      setNotice(error.message || 'Не удалось обработать файлы.', true);
      if (/503|not configured/i.test(error.message || '')) checkReady();
    } finally { state.busy = false; updateUpload(); }
  }
  input.addEventListener('change', () => { state.files = [...input.files]; updateUpload(); });
  byId('upload-form').addEventListener('submit', event => { event.preventDefault(); run(state.files); });
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
  dropzone.addEventListener('drop', event => { state.files = [...event.dataTransfer.files]; input.value = ''; updateUpload(); });
  byId('search').addEventListener('input', () => { if (state.result) renderRows(); });
  byId('retry-ready').addEventListener('click', checkReady);
  byId('download-csv').addEventListener('click', () => {
    if (!state.result) return;
    const blob = new Blob([state.result.csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = node('a'); link.href = url; link.download = 'results.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  checkReady();
})();
