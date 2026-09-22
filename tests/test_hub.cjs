'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../docs/assets/hub.js'), 'utf8');

function element() {
  return {
    textContent: '',
    attributes: {},
    get innerHTML() {
      return this.textContent.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    },
    set innerHTML(value) {
      assert.fail(`Error messages must use textContent, received HTML: ${value}`);
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
  };
}

async function loadHub({ missingLibrary, httpStatus = 200, parsed, fetchError } = {}) {
  const status = element();
  const renderer = {};
  const tables = [];
  let fetches = 0;
  const row = { '<heading>': '<img src=x onerror=alert(1)>', 'metric.with.dots': '2.5' };

  function DataTable(selector, options) {
    assert.equal(selector, '#exp-table');
    tables.push(options);
  }
  DataTable.render = { text: () => renderer };

  const context = {
    document: {
      getElementById: () => status,
      createElement: element,
    },
    console: { error() {} },
    fetch: async () => {
      fetches += 1;
      if (fetchError) throw fetchError;
      return { ok: httpStatus === 200, status: httpStatus, text: async () => 'fixture' };
    },
    DataTable,
    Papa: {
      parse: () => parsed ?? { data: [row], meta: { fields: Object.keys(row) }, errors: [] },
    },
  };
  if (missingLibrary) delete context[missingLibrary];

  await vm.runInNewContext(source, context, { timeout: 1000 });
  return { status, renderer, tables, fetches, row };
}

test('renders source headings as text and delegates cell escaping to DataTables', async () => {
  const { status, renderer, tables, row } = await loadHub();
  assert.equal(tables.length, 1);
  assert.equal(tables[0].columns[0].title, '&lt;heading&gt;');
  assert.equal(tables[0].columns[0].render, renderer);
  assert.equal(tables[0].columns[0].data(row), '<img src=x onerror=alert(1)>');
  assert.equal(tables[0].columns[1].data(row), '2.5');
  assert.match(status.textContent, /1 historical experiment/);
  assert.equal(status.attributes['data-error'], undefined);
});

test('reports missing dependencies without fetching data or retrying indefinitely', async () => {
  for (const missingLibrary of ['Papa', 'DataTable']) {
    const { status, tables, fetches } = await loadHub({ missingLibrary });
    assert.equal(fetches, 0);
    assert.equal(tables.length, 0);
    assert.equal(status.attributes.role, 'alert');
    assert.match(status.textContent, /libraries could not load/);
  }
});

test('reports HTTP and network failures without inserting error HTML', async () => {
  for (const scenario of [{ httpStatus: 404 }, { fetchError: new Error('<img src=x onerror=alert(1)>') }]) {
    const { status, tables } = await loadHub(scenario);
    assert.equal(tables.length, 0);
    assert.equal(status.attributes.role, 'alert');
    assert.equal(status.attributes['data-error'], '');
    assert.equal(status.textContent, scenario.fetchError?.message ?? 'Source data request failed (HTTP 404).');
  }
});

test('rejects malformed or empty parsed tables before constructing DataTables', async () => {
  const cases = [
    { data: [{ loss: '2.5' }], meta: { fields: ['loss'] }, errors: [{ code: 'TooFewFields' }] },
    { data: [], meta: { fields: ['loss'] }, errors: [] },
    { data: [{ loss: '2.5' }], meta: {}, errors: [] },
  ];
  for (const parsed of cases) {
    const { status, tables } = await loadHub({ parsed });
    assert.equal(tables.length, 0);
    assert.equal(status.attributes.role, 'alert');
    assert.match(status.textContent, /empty or malformed/);
  }
});
