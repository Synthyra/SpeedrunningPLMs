(async () => {
  const status = document.getElementById('load-status');
  const sourceUrl = 'https://raw.githubusercontent.com/Synthyra/SpeedrunningPLMs/main/misc/experiments.tsv';

  try {
    if (typeof Papa === 'undefined' || typeof DataTable === 'undefined') {
      throw new Error('Table libraries could not load. Reload the page or open the source data.');
    }

    const response = await fetch(sourceUrl);
    if (!response.ok) {
      throw new Error(`Source data request failed (HTTP ${response.status}).`);
    }

    const { data, meta, errors } = Papa.parse(await response.text(), {
      delimiter: '\t',
      header: true,
      skipEmptyLines: 'greedy',
      transformHeader: header => header.trim(),
    });
    if (errors.length || !meta.fields?.length || !data.length) {
      throw new Error('The source table is empty or malformed. Open the source data for details.');
    }

    const columns = meta.fields.map(field => {
      const title = document.createElement('span');
      title.textContent = field;
      return {
        title: title.innerHTML,
        data: row => row[field],
        defaultContent: '',
        render: DataTable.render.text(),
      };
    });
    new DataTable('#exp-table', {
      data,
      columns,
      searching: true,
      ordering: true,
      paging: true,
      pageLength: 25,
      order: [],
    });
    status.textContent = `${data.length} historical experiments loaded.`;
  } catch (error) {
    console.error('Could not load historical experiments:', error);
    status.textContent = error.message;
    status.setAttribute('data-error', '');
    status.setAttribute('role', 'alert');
  }
})();
