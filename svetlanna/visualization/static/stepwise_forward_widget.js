function render({ model, el }) {

    let container = document.createElement("div")
    let containerStructure = document.createElement("div")
    let containerImages = document.createElement("div")

    container.appendChild(containerStructure)
    container.appendChild(containerImages)

    let elements = model.get("elements")
    let structureHTML = model.get("structure_html")

    containerStructure.className = 'svetlanna-structure-container'
    const containerImagesInnerHTML = `
    <div id="img-container" class="svetlanna-specs-container" style="font-family: monospace;">
        Click on any element
    </div>
    `

    containerStructure.innerHTML = structureHTML
    containerImages.innerHTML = containerImagesInnerHTML

    function insertElementSpecsHtml(el, element) {
        let html = `<div style="display:block;margin-bottom:6px;"><b>(${element.index}) ${element.name}</b> output:</div>`

        if (element.output_image === null) {
            html += '<p>No output has been captured</p>'
        } else if (element.output_image.startsWith('\n')) {
            html += `<p>During the calculations or plotting exception has been raised:${element.output_image}</p>`
        } else {
            html += `<img src="data:image/png;base64,${element.output_image}" style="object-fit: contain; max-width: 100%; height: auto; width: auto; display: block;">`
        }

        el.querySelector('#img-container').innerHTML = html
    }

    el.appendChild(container);
    elements.forEach(element => {
        let elementDiv = containerStructure.querySelector(`#sv-${element.index}`)
        elementDiv.onclick = (event) => {
            insertElementSpecsHtml(containerImages, element)
            event.stopPropagation()
            event.preventDefault()
        }
    })
}

export default { render };
