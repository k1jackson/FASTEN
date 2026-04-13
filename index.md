---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: FASTEN Configuration File Designer
---

## Data Parameters
<div class="upload-container" style="padding-bottom: 20px;">
    <label for="tsv-upload" style="display: inline-block; min-width: 250px;"><strong>Training Data File</strong> <i style="color: #1e6bb8;">(.tsv)</i></label>
    <input type="file" id="tsv-upload" accept=".tsv" style="margin-right: 5px">
</div>
<div id="data-params" style="display: none; min-width: 200px;"></div>

## Model Parameters
<div class="export-container" id="export-options" style="padding-bottom: 20px;">
    <span style="display: inline-block; min-width: 250px"><strong>Configuration File</strong> <i style="color: #1e6bb8;">(.json)</i></span>
    <label style="margin-right: 20px;"><input type="radio" style="margin-right: 5px" class="export-option" name="model-mode" value="Train">Train</label>
    <label style="margin-right: 20px;"><input type="radio" style="margin-right: 5px" class="export-option" name="model-mode" value="Tune">Tune</label>
    <button id="export-button" style="display: none; margin-left: 25px;">Export</button>
</div>
<div id="model-params" style="display: none; min-width: 200px;">
    <p>hello</p>
</div>

<script>
const dataContainer = document.getElementById('data-params');
const modelContainer = document.getElementById('model-params');
const exportButton = document.getElementById('export-button');
let has_file = false;
let export_mode = "";
const modelConfig = [
    {
        group: "Training Data Split Proportions",
        params: [
            { id: "test-split", train_placeholder: "0", tune_placeholder: "0",
              train_label: 'Testing Set Proportion:<br><i style="color: #1e6bb8;">(non-negative float)</i>', tune_label: 'Testing Set Proportion:<br><i style="color: #1e6bb8;">(non-negative float)</i>',
              train_alert: '- Invalid testing set proportion. Must be non-negative float between 0 (inclusive) and 1.',
              tune_alert: '- Invalid testing set proportion. Must be non-negative float between 0 (inclusive) and 1. Not a tunable parameter.' },
            { id: "valid-split", train_placeholder: "0", tune_placeholder: "0",
              train_label: 'Validation Set Proportion:<br><i style="color: #1e6bb8;">(non-negative float)</i>', tune_label: 'Validation Set Proportion:<br><i style="color: #1e6bb8;">(non-negative float)</i>',
              train_alert: '- Invalid validation set proportion. Must be non-negative float between 0 (inclusive) and 1.',
              tune_alert: '- Invalid validation set proportion. Must be non-negative float between 0 (inclusive) and 1. Not a tunable parameter.' }
        ]
    },
    {
        group: "Neural Network Parameters",
        params: [
            { id: "architecture", type: "select", options: ["Pyramidal", "Rectangular"],
              train_label: 'Architecture:', tune_label: 'Architecture:<br><i style="color: #1e6bb8;">(select one or more)</i>' },
            { id: "hidden-layers", train_placeholder: "2", tune_placeholder: "2, 4, 6, 8",
              train_label: 'Number of Hidden Layers:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              train_alert: '- Invalid number of hidden layers. Must be positive integer.',
              tune_label: 'Number of Hidden Layers:<br><i style="color: #1e6bb8;">(list of positive integers)</i>',
              tune_alert: '- Invalid number(s) of hidden layers. Must be list of positive integers.' },
            { id: "hidden-size", train_placeholder: "64", tune_placeholder: "32, 64, 128, 256, 512",
              train_label: 'Max Hidden Layer Size:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              train_alert: '- Invalid hidden layer size. Must be positive integer.',
              tune_label: 'Max Hidden Layer Size:<br><i style="color: #1e6bb8;">(list of positive integers)</i>',
              tune_alert: '- Invalid hidden layer size(s). Must be list of positive integers.' }
        ]
    },
    {
        group: "Training Loop Parameters",
        params: [
            { id: "device", type: "select", options: ["CPU", "CUDA"], 
              train_label: 'Processing Device:', tune_label: 'Processing Device:' },
            { id: "batch-size", train_placeholder: "64", tune_placeholder: "32, 64, 128, 256, 512", 
              train_label: 'Batch Size:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              train_alert: '- Invalid batch size. Must be positive integer.',
              tune_label: 'Batch Size:<br><i style="color: #1e6bb8;">(list of positive integers)</i>',
              tune_alert: '- Invalid batch size(s). Must be list of positive integers.' },
            { id: "num-epochs", train_placeholder: "5000", tune_placeholder: "5000",
              train_label: 'Max Number of Epochs:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              train_alert: '- Invalid number of epochs. Must be positive integer.',
              tune_label: 'Max Number of Epochs:<br><i style="color: #1e6bb8;">(list of positive integers)</i>',
              tune_alert: '- Invalid number(s) of epochs. Must be list of positive integers.' },
            { id: "early-stop", type: "check", train_label: 'Early Stopping:', tune_label: 'Early Stopping:' },
            { id: "patience", train_placeholder: "20", tune_placeholder: "20",
              train_label: 'Patience:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              train_alert: '- Invalid patience. Must be positive integer.',
              tune_label: 'Patience:<br><i style="color: #1e6bb8;">(positive integer)</i>',
              tune_alert: '- Invalid patience. Must be positive integer.' },
            { id: "min-delta", train_placeholder: "0", tune_placeholder: "0",
              train_label: 'Min Delta:<br><i style="color: #1e6bb8;">(float)</i>',
              train_alert: '- Invalid minimum delta. Must be float.',
              tune_label: 'Min Delta:<br><i style="color: #1e6bb8;">(float)</i>',
              tune_alert: '- Invalid minimum delta. Must be float.' },
        ]
    },
    {
        group: "Optimizer Parameters",
        params: [
            { id: "optimizer", type: "select", options: ["SGD", "Adam", "AdamW"], 
              train_label: 'Optimization Algorithm:', tune_label: 'Optimization Algorithm:<br><i style="color: #1e6bb8;">(select one or more)</i>' },
            { id: "learning-rate", train_placeholder: "0.001", tune_placeholder: "0.001, 0.0005, 0.0001, 0.00005",
              train_label: 'Learning Rate:<br><i style="color: #1e6bb8;">(positive float)</i>',
              train_alert: '- Invalid learning rate. Must be positive float.',
              tune_label: 'Learning Rate:<br><i style="color: #1e6bb8;">(list of positive floats)</i>',
              tune_alert: '- Invalid learning rate(s). Must be list of positive floats.' },
            { id: "weight-decay", train_placeholder: "0", tune_placeholder: "0",
              train_label: 'Weight Decay:<br><i style="color: #1e6bb8;">(non-negative float)</i>',
              train_alert: '- Invalid weight decay. Must be non-negative float.',
              tune_label: 'Weight Decay:<br><i style="color: #1e6bb8;">(list of non-negative floats)</i>',
              tune_alert: '- Invalid weight decay(s). Must be list of non-negative floats.' },
            { id: "momentum", train_placeholder: "0", tune_placeholder: "0",
              train_label: 'Momentum:<br><i style="color: #1e6bb8;">(non-negative float)</i>',
              train_alert: '- Invalid momentum. Must be non-negative float.',
              tune_label: 'Momentum:<br><i style="color: #1e6bb8;">(list of non-negative floats)</i>',
              tune_alert: '- Invalid momentum(s). Must be list of non-negative floats.'}
        ]
    }
];
function renderModelParams(export_mode) {
    modelContainer.innerHTML = ''; 
    modelConfig.forEach(section => {
        const sectionRow = document.createElement('div');
        sectionRow.className = "model-param-group";
        sectionRow.style.marginBottom = "10px";
        sectionRow.style.borderBottom = "1px solid #eee"; 
        sectionRow.style.paddingBottom = "10px";

        const header = document.createElement('div');
        header.innerHTML = `<span style="display: inline-block; min-width: 200px;"><strong>${section.group}</strong></span>`;
        sectionRow.appendChild(header);

        section.params.forEach(param => {
            const row = document.createElement('div');
            row.id = `${param.id}-row`;
            row.style.marginTop = "10px";
            row.style.paddingLeft = "20px";
            row.style.display = "flex";
            row.style.alignItems = "flex-start";

            let inputHtml = '';
            if (param.type === 'select') {
                if (param.id === 'architecture' && export_mode == 'Tune') {
                    inputHtml = `<select id="${param.id}" multiple required size="2"}>${param.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}</select>`;
                } else if (param.id === 'optimizer' && export_mode == 'Tune') {
                    inputHtml = `<select id="${param.id}" multiple required size="3">${param.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}</select>`;
                } else { inputHtml = `<select id="${param.id}">${param.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}</select>`; }
            } else if (param.type === 'check') {
                inputHtml = `<div display="display: flex; align-items: center;"><input id="${param.id}" type="checkbox" checked></div>`;
            } else {
                if (export_mode === 'Tune' && param.id != 'test-split' && param.id != 'valid-split' && param.id != 'patience' && param.id != 'min-delta') { inputHtml = `<input type="text" id="${param.id}" style="width: 400px" placeholder="${param.tune_placeholder}">`; }
                else if (export_mode === 'Tune') { inputHtml = `<input type="text" id="${param.id}" style="width: 100px" placeholder="${param.tune_placeholder}">`; }
                else { inputHtml = `<input type="text" id="${param.id}" style="width: 100px" placeholder="${param.train_placeholder}">`; }
            }
            if (export_mode === 'Train') { row.innerHTML = `<label for="${param.id}" style="display: inline-block;  min-width: 220px; text-align: right; margin-right: 10px">${param.train_label}</label></div>${inputHtml}`; }
            if (export_mode === 'Tune') { row.innerHTML = `<label for="${param.id}" style="display: inline-block; min-width: 220px; text-align: right; margin-right: 10px">${param.tune_label}</label>${inputHtml}`; }
            sectionRow.appendChild(row);
        });
        modelContainer.appendChild(sectionRow);
    });
    const optimizerSelect = document.getElementById('optimizer');
    const architectureSelect = document.getElementById('architecture');
    const earlyStopSelect = document.getElementById('early-stop');
    const momentumRow = document.getElementById('momentum-row');
    optimizerSelect.options[0].selected = true;
    architectureSelect.options[0].selected = true;

    if (optimizerSelect && momentumRow) {
        const toggleMomentum = () => {
            momentumRow.style.display = (optimizerSelect.value === 'SGD') ? 'flex' : 'none';
        };
        toggleMomentum();
        optimizerSelect.addEventListener('change', toggleMomentum);
    }
    earlyStopSelect.addEventListener('change', function(event) {
        const patience = document.getElementById('patience-row')
        const min_delta = document.getElementById('min-delta-row')
        if (event.target.checked) {
            patience.style.display = 'flex';
            min_delta.style.display = 'flex';
        } else {
            patience.style.display = 'none';
            min_delta.style.display = 'none';
        }
    });
}
dataContainer.addEventListener('change', function(event) {
    if (event.target.classList.contains('param-option')) {
        const radio = event.target;
        const parentRow = radio.closest('.data-param');
        const typeOptions = parentRow.querySelector('.type-options');
        const distOptions = parentRow.querySelector('.dist-options');
        const nameEntry = parentRow.querySelector('.name-entry');
        const minThreshEntry = parentRow.querySelector('.min-thresh-entry');
        const maxThreshEntry = parentRow.querySelector('.max-thresh-entry');


        if (radio.checked) {
            if (radio.value === 'input') {
                nameEntry.style.display = 'block';
                typeOptions.style.display = 'block';
                minThreshEntry.style.display = 'none';
                maxThreshEntry.style.display = 'none';
                distOptions.style.display = 'none';
            } else if (radio.value === 'output') {
                nameEntry.style.display = 'block';
                typeOptions.style.display = 'block';
                minThreshEntry.style.display = 'block';
                maxThreshEntry.style.display = 'block';
                distOptions.style.display = 'block';
            } else {
                nameEntry.style.display = 'none';
                typeOptions.style.display = 'none';
                minThreshEntry.style.display = 'none';
                maxThreshEntry.style.display = 'none';
                distOptions.style.display = 'none';
            }
        } else {
            nameEntry.style.display = 'none';
            typeOptions.style.display = 'none';
            minThreshEntry.style.display = 'none';
            maxThreshEntry.style.display = 'none';
            distOptions.style.display = 'none';
        }
    }
});
document.getElementById('tsv-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];

    if (file) {
        const fileName = file.name; 
        const fileExtension = fileName.split('.').pop().toLowerCase();
        if (fileExtension !== 'tsv') {
            alert("Invalid Training Data File Type.");
            event.target.value = '';
            return; 
        } 
        has_file = true;
        dataContainer.innerHTML = '';
        dataContainer.style.display = 'block';
        const reader = new FileReader();
        reader.onload = function(e) {
            const lines = e.target.result.split(/\r?\n/);
            if (lines.length > 0) {
                const headers = lines[0].split('\t');
                headers.forEach((header, index) => {
                    const clean_header = header.trim();
                    if (clean_header === "") return; 

                    const row = document.createElement('div');
                    row.className = "data-param"
                    row.style.marginBottom = "10px";
                    row.style.borderBottom = "1px solid #eee";
                    row.style.paddingBottom = "10px";

                    row.innerHTML = `
                        <span style="display: inline-block; min-width: 300px;"><strong>${clean_header}</strong></span>
                        <label style="margin-right: 20px;"><input type="radio" style="margin-right: 5px" class="param-option" name="group-${index}" value="input">Input</label>
                        <label style="margin-right: 20px;"><input type="radio" style="margin-right: 5px" class="param-option" name="group-${index}" value="output">Output</label>
                        <label style="margin-right: 20px;"><input type="radio" style="margin-right: 5px" class="param-option" name="group-${index}" value="exclude" checked>Exclude</label>

                        
                        <div id="name-input-${index}" class="name-entry" style="display: none; margin-top: 10px; padding-left: 20px;">
                        <label for="name-${index}" style="margin-right: 5px;">Name:</label>
                        <input type="text" id="name-${index}" style="width: 400px" name="name" placeholder="${clean_header}">
                        </div>

                        <div id="min-thresh-input-${index}" class="min-thresh-entry" style="display: none; margin-top: 10px; padding-left: 20px;">
                        <label for="min-thresh-${index}" style="margin-right: 5px;">Min Threshold:</label>
                        <input type="text" id="min-thresh-${index}" style="width: 100px" name="min-thresh">
                        </div>

                        <div id="max-thresh-input-${index}" class="max-thresh-entry" style="display: none; margin-top: 10px; padding-left: 20px;">
                        <label for="max-thresh-${index}" style="margin-right: 5px;">Max Threshold:</label>
                        <input type="text" id="max-thresh-${index}" style="width: 100px" name="max-thresh">
                        </div>

                        <div id="type-select-${index}" class="type-options" style="display: none; margin-top: 10px; padding-left: 20px;">
                        <label for="type-${index}" style="margin-right: 5px;"><span style="color: red;">*</span> Data Type:</label>
                        <select id="type-${index}">
                            <option value="float" selected>Float</option>
                            <option value="int">Integer</option>
                            <option value="string">String</option>
                        </select>
                        </div>

                        <div id="dist-select-${index}" class="dist-options" style="display: none; margin-top: 10px; padding-left: 20px;">
                        <label for="dist-${index}" style="margin-right: 5px;"><span style="color: red;">*</span> Distribution:</label>
                        <select id="dist-${index}">
                            <option value="Beta">Beta</option>
                            <option value="Exponential">Exponential</option>
                            <option value="Gamma">Gamma</option>
                            <option value="Laplace">Laplace</option>
                            <option value="LogNormal">LogNormal</option>
                            <option value="Normal" selected>Normal</option>
                            <option value="Uniform">Uniform</option>
                        </select>
                        </div>
                    `;

                    dataContainer.appendChild(row);
                });
            }
        };
        reader.readAsText(file);
    } else { 
        return; 
    }
});
document.getElementById('export-options').addEventListener('change', function(event) {
    if (event.target.classList.contains('export-option')) {
        if (event.target.checked) {
            renderModelParams(event.target.value); 
            export_mode = event.target.value;
            modelContainer.style.display = 'block';
            exportButton.style.display = 'inline-block';
        }
    }
});
function getFieldValue(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (id === 'early-stop') { return el.checked }
    const isHidden = el.closest('div').style.display === 'none';
    if (isHidden || el.value.trim() === "") {
        return el.placeholder || el.value; 
    }
    if (export_mode === 'Tune' && (id === 'architecture' || id === 'optimizer')) {
        return Array.from(el.selectedOptions).map(opt => opt.value);
    }
    return el.value;
}

function collectAndExport() {
    const integers = ["hidden-layers", "hidden-size", "batch-size", "num-epochs", "patience"]
    const nn_floats = ["learning-rate", "weight-decay", "momentum", "min-delta"]
    const props = ["test-split", "valid-split"]
    const tunable = ["hidden-layers", "hidden-size", "batch-size", "num-epochs", "learning-rate", "weight-decay", "momentum"]
    const config = {
        inputs: {},
        outputs: {},
        params: {}
    };
    let is_valid = true;
    let alert_msg = [];
    let input_counts = 0;
    let output_counts = 0;

    if (!has_file) {
        is_valid = false;
        alert_msg.push("- Invalid data parameters. Missing training data file.")
    }

    const dataRows = document.querySelectorAll('.data-param');
    dataRows.forEach((row, index) => {
        const headerName = row.querySelector('strong').innerText;
        const selectedOption = row.querySelector('input[type="radio"]:checked');
        const role = selectedOption ? selectedOption.value : 'exclude';
        if (role === 'input') {
            config.inputs[headerName] = {
                name: getFieldValue(`name-${index}`),
                type: document.getElementById(`type-${index}`).value
            };
            input_counts += 1;
        }
        if (role === 'output') {
            const min_thresh = document.getElementById(`min-thresh-${index}`);
            const max_thresh = document.getElementById(`max-thresh-${index}`);
            let min_thresh_value = null;
            let max_thresh_value = null;
            if (min_thresh.value !== undefined) {
                min_thresh_value = Number(min_thresh.value);
                if (isNaN(min_thresh_value)) {
                    is_valid = false;
                    alert_msg.push("- Invalid minimum threshold. Must be a number.");
                }
            }
            if (max_thresh.value !== undefined) {
                max_thresh_value = Number(max_thresh.value);
                if (isNaN(max_thresh_value)) {
                    is_valid = false;
                    alert_msg.push("- Invalid minimum threshold. Must be a number.");
                }
            }
            config.outputs[headerName] = {
                name: getFieldValue(`name-${index}`),
                min_threshold: min_thresh_value,
                max_threshold: max_thresh_value,
                type: document.getElementById(`type-${index}`).value,
                distribution: document.getElementById(`dist-${index}`).value
            };
            output_counts += 1;
        }
    });
    if (input_counts == 0) {
        is_valid = false;
        alert_msg.push("- Invalid input parameters. Requires at least one input.")
    }
    if (output_counts == 0) {
        is_valid = false;
        alert_msg.push("- Invalid output parameters. Requires at least one output.")
    }
    if (export_mode == "Train") {
        modelConfig.forEach(section => {
            section.params.forEach(param => {
                const temp = getFieldValue(param.id);
                const param_id = param.id.replace("-", "_");
                const num = Number(temp);
                if (param.id === 'device' || param.id === 'architecture') {
                    config.params[param_id] = temp.toLowerCase();
                } else if (integers.includes(param.id)) {
                    if (Number.isInteger(num) && num > 0) {
                        config.params[param_id] = num;
                    } else {
                        alert_msg.push(param.train_alert);
                        is_valid = false;
                    }
                } else if (nn_floats.includes(param.id)) {
                    if (!isNaN(num) && param.id === 'learning-rate' && num > 0) {
                        config.params[param_id] = num;
                    } else if (!isNaN(num) && param.id === 'min-delta') {
                        config.params[param_id] = num;
                    } else if (!isNaN(num) && param.id !== 'learning-rate' && param.id !== 'min-delta' && num >= 0) {
                        config.params[param_id] = num;
                    } else {
                        alert_msg.push(param.train_alert);
                        is_valid = false;
                    }
                } else if (props.includes(param.id)) {
                    if (!isNaN(num) && 0 <= num && num < 1) {
                        config.params[param_id] = num;
                    } else {
                        alert_msg.push(param.train_alert);
                        is_valid = false;
                    }
                } else {
                    config.params[param_id] = temp;
                }
            });
        });
    } else if (export_mode == "Tune") {
        modelConfig.forEach(section => {
            section.params.forEach(param => {
                const temp = getFieldValue(param.id);
                const param_id = param.id.replace("-", "_");
                let is_list = false;
                try { is_list = /^[^,]+(,[^,]+)+$/.test(temp.trim()); }
                catch (error) { }
                if (is_list && tunable.includes(param.id)) {
                    const items = temp.split(',').map(item => item.trim());
                    let item_nums = [];
                    let valid_items = true;
                    items.forEach(item => {
                        const num = Number(item);
                        if (integers.includes(param.id)) {
                            if (Number.isInteger(num) && num > 0) {
                                item_nums.push(num)
                            } else {
                                valid_items = false;
                            }
                        } else if (nn_floats.includes(param.id)) {
                            if (!isNaN(num) && param.id === 'learning-rate' && num > 0) {
                                item_nums.push(num)
                            } else if (!isNaN(num) && param.id !== 'learning-rate' && num >= 0) {
                                item_nums.push(num)
                            } else {
                                valid_items = false;
                            }
                        }
                    });
                    if (!valid_items) {
                        alert_msg.push(param.tune_alert);
                        is_valid = false;
                    } else {
                        config.params[param_id] = item_nums
                    }
                } else {
                    const num = Number(temp);
                    if (param.id === 'device') {
                        config.params[param_id] = temp.toLowerCase();
                    } else if (param.id === 'architecture') {
                        config.params[param_id] = Array.from(temp).map(opt => opt.toLowerCase());
                    } else if (integers.includes(param.id)) {
                        if (Number.isInteger(num) && num > 0) {
                            config.params[param_id] = num;
                        } else {
                            alert_msg.push(param.tune_alert);
                            is_valid = false;
                        }
                    } else if (nn_floats.includes(param.id)) {
                        if (!isNaN(num) && param.id === 'learning-rate' && num > 0) {
                            config.params[param_id] = num;
                        } else if (!isNaN(num) && param.id !== 'learning-rate' && num >= 0) {
                            config.params[param_id] = num;
                        } else {
                            alert_msg.push(param.tune_alert);
                            is_valid = false;
                        }
                    } else if (props.includes(param.id)) {
                        if (!isNaN(num) && 0 <= num && num < 1) {
                            config.params[param_id] = num;
                        } else {
                            alert_msg.push(param.tune_alert);
                            is_valid = false;
                        }
                    } else {
                        config.params[param_id] = temp;
                    }
                }
            });
        });
    }
    if (!is_valid) {
        alert(alert_msg.join("\n"));
        return; 
    }

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(config, null, 4));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "config.json");
    document.body.appendChild(downloadAnchorNode); 
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}
exportButton.addEventListener('click', collectAndExport);
</script>
