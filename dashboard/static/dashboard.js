const aggregateInput = document.getElementById("aggregate_input");
const stepInput = document.getElementById("step_input");

function submitForm() {
    // If aggregate input is empty, disable its name
    // so it will not appear in the form
    if (aggregateInput.value == "") {
        aggregateInput.name = "";
    }

    // Same for step input
    if (stepInput.value == "") {
        stepInput.name = "";
    }

    document.getElementById("form_select_sweep").submit();
}

let interactiveElements = Array.from(document.getElementsByClassName("interactive"));
interactiveElements.forEach(element => {
    element.onchange = () => submitForm();
});

function doAggregate(parameter) {
    aggregateInput.value = parameter;
    submitForm();
}

let slideStopTimer = null;
let stepSlider = document.getElementById("step_slider");
let selectedStep = document.getElementById("selected_step");
if (stepSlider != null) {
    let sliderContainer = document.getElementById("slider_container");
    let sliderWidth = 500 - 20;
    console.log(sliderWidth);
    stepSlider.style.width = `${sliderWidth}px`;

    let slider = new Slider("#step_slider");
    slider.on("change", (slideInfo) => {
        selectedStep.innerHTML = slideInfo.newValue;
    });

    slider.on("slideStop", (newValue) => {
        if (slideStopTimer != null) {
            clearTimeout(slideStopTimer);
        }

        slideStopTimer = setTimeout(() => {
            stepInput.value = newValue;
            submitForm();
        }, 250);
    });
}