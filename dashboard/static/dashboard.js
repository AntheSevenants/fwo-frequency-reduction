const aggregateInput = document.getElementById("aggregate_input");

function submitForm() {
    // If aggregate input is empty, disable its name
    // so it will not appear in the form
    if (aggregateInput.value == "") {
        aggregateInput.name = "";
    }
    document.getElementById("form_select_run").submit();
}

let interactiveElements = Array.from(document.getElementsByClassName("interactive"));
interactiveElements.forEach(element => {
    element.onchange = () => submitForm();
});

function doAggregate(parameter) {
    aggregateInput.value = parameter;
    submitForm();
}