let interactiveElements = Array.from(document.getElementsByClassName("interactive"));
interactiveElements.forEach(element => {
    element.onchange = () => document.getElementById("form_select_run").submit();
});