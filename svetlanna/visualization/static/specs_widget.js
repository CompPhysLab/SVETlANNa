function render({ model, el }) {

    let container = document.createElement("div")
    let containerStructure = document.createElement("div")
    let containerSpecs = document.createElement("div")

    container.appendChild(containerStructure)
    container.appendChild(containerSpecs)

    let elements = model.get("elements")
    let structureHTML = model.get("structure_html")

    containerStructure.className = 'svetlanna-structure-container'
    const containerSpecsInnerHTML = `
    <div class="svetlanna-specs-container">
    <details open>
        <summary id="specs-summary">
            Click on any element
        </summary>
        <div class="svetlanna-specs-element-block" id="specs">
        </div>
    </details>
    </div>
    `

    containerStructure.innerHTML = structureHTML
    containerSpecs.innerHTML = containerSpecsInnerHTML

    function insertElementSpecsHtml(el, element) {
        el.querySelector('#specs-summary').textContent = `(${element.index}) ${element.name}`
        el.querySelector('#specs').innerHTML = element.specs_html
    }
    
    el.appendChild(container);

    elements.forEach(element => {
        let elementDiv = containerStructure.querySelector(`#sv-${element.index}`)
        elementDiv.onclick = (event) => {
            insertElementSpecsHtml(containerSpecs, element)
            event.stopPropagation()
            event.preventDefault()
        }
    })
}

export default { render };
