function render({ model, el }) {

let container = document.createElement("div")
let containerElements = document.createElement("div")
let containerSpecs = document.createElement("div")

container.appendChild(containerElements)
container.appendChild(containerSpecs)

let elements = model.get("elements")
let settings = model.get("settings")

containerElements.className = 'svetlanna-elements-el-container'
const containerSpecsInnerHTML = `
<div class="svetlanna-specs-container">
<details ${settings.open?'open':''}>
<summary id="specs-summary">Click on any element</summary>
    <div class="svetlanna-specs-details" id="specs">
    </div>
</details>
</div>
`

if (!settings.show_all) {
    containerSpecs.innerHTML = containerSpecsInnerHTML
}

function insertElementSpecsHtml(el, element) {
    el.querySelector('#specs-summary').textContent = `(${element.index}) ${element.type}`
    el.querySelector('#specs').innerHTML = element.specs_html
}

elements.forEach(element => {
    let elementDiv = document.createElement("div")
    elementDiv.className = 'svetlanna-element clickable'
    elementDiv.classList.add(element.type)
    elementDiv.innerHTML = `<div class="svetlanna-element-index clickable">(${element.index}) ${element.type}</div>`
    containerElements.appendChild(elementDiv)

    if (!settings.show_all) {
        elementDiv.onclick = () => {
            insertElementSpecsHtml(containerSpecs, element)
        }
    } else {
        let elementContainer = document.createElement("div")
        containerSpecs.appendChild(elementContainer)
        elementContainer.innerHTML = containerSpecsInnerHTML
        insertElementSpecsHtml(elementContainer, element)
    }

});

el.appendChild(container);
}

export default { render };
