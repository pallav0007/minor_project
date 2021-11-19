let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');

const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
    const files = fileInput.files;

    console.log(fileInput.files);
    [...files].map(async (img) => {
        const data = new FormData();
        data.append('file', img);
        console.log(data)
        const processedImage = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => {
                return result['image'];
            });

        // shape has to be the same as it was for training of the model
        const prediction = model.predict(tf.reshape(processedImage, shape = [1, 100,100, 1]));
        const characters=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
        const label = characters[prediction.argMax(axis = 1).dataSync()[0]];
        renderImageLabel(img, label);
    })
};

const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    reader.onload = () => {
        preview.innerHTML += `<div style="width: 40%;height: 40%" class="image-block">
                                      <img src="${reader.result}" class="image-block_loaded" id="source"/>
                                       <h2 class="image-block__label">${label}</h2>
                              </div>`;

    };
    reader.readAsDataURL(img);
};


fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");
