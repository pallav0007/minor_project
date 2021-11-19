let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');
function ShowCam() {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 100
    });
    Webcam.attach('#my_camera');
}
window.onload= ShowCam;

function dataURItoBlob(dataURI) {
  // convert base64 to raw binary data held in a string
  // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
  var byteString = atob(dataURI.split(',')[1]);

  // separate out the mime component
  var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]

  // write the bytes of the string to an ArrayBuffer
  var ab = new ArrayBuffer(byteString.length);

  // create a view into the buffer
  var ia = new Uint8Array(ab);

  // set the bytes of the buffer to the correct values
  for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
  }

  // write the ArrayBuffer to a blob, and you're done
  var blob = new Blob([ab], {type: mimeString});
  return blob;

}
var datauri=null;
        function snap() {
    Webcam.snap( function(data_uri) {
        // display results in page
        datauri=data_uri;
        var blob = dataURItoBlob(data_uri);
        const form = new FormData();
        form.append("file", blob);
        console.log(form)
        document.getElementById('results').innerHTML =
        '<img id="image" src="'+data_uri+'"/>';
      } );
}
const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
        var image = datauri;
        var img = dataURItoBlob(image);

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

};

const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    reader.onload = () => {
        preview.innerHTML += `<div class="image-block">
                                      <img src="${reader.result}" class="image-block_loaded" id="source"/>
                                       <h2 class="image-block__label">${label}</h2>
                              </div>`;

    };
    reader.readAsDataURL(img);
};


fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");
