async function onWindowLoad() {

    const canvasLatent = document.getElementById("latent-canvas");
    canvasLatent.width = 4;
    canvasLatent.height = 4;
    const contextLatent = canvasLatent.getContext("2d");
    contextLatent.clearRect(0, 0, 4, 4);

    const canvasDecoded = document.getElementById("decoded-canvas");
    canvasDecoded.width = 28;
    canvasDecoded.height = 28;
    const contextDecoded = canvasDecoded.getContext("2d");
    contextDecoded.clearRect(0, 0, 28, 28);
  
    const decoder = await tf.loadGraphModel("decoder/model.json");

    const latentBase = []; // 10 latent vectors, one per digit
    const minValues = []; // minimum value for each of the 16 latent dimension
    const maxValues = []; // maximum value for each of the 16 latent dimension

    const medianVectors = await (fetch("median.json")).then(res => res.json()); 
    for (let k = 0; k < medianVectors.length; k++) {
        latentBase.push(tf.tensor(medianVectors[k]).reshape([1, 2, 2, 4]));
    }
    for (let p = 0; p < 16; p++) {
        minValues.push(null);
        maxValues.push(null);
    }
    for (let k = 0; k < latentBase.length; k++) {
        const arr = latentBase[k].reshape([16]).arraySync();
        for (let p = 0; p < 16; p++) {
            if (minValues[p] == null || arr[p] < minValues[p]) {
                minValues[p] = arr[p];
            }
            if (maxValues[p] == null || arr[p] > maxValues[p]) {
                maxValues[p] = arr[p];
            }
        }
    }

    function drawLatentTensor(tensor) {
        const array = tensor.reshape([16]).arraySync();
        const imageData = contextLatent.createImageData(4, 4);
        for (let p = 0; p < 16; p++) {
            const grey = 255 * (array[p] - minValues[p]) / (maxValues[p] - minValues[p]);
            imageData.data[p * 4] = grey;
            imageData.data[p * 4 + 1] = grey;
            imageData.data[p * 4 + 2] = grey;
            imageData.data[p * 4 + 3] = 255;
        }
        contextLatent.putImageData(imageData, 0, 0);
    }

    function drawDecodedTensor(tensor) {
        const array = tensor.reshape([28, 28]).arraySync();
        const imageData = contextDecoded.createImageData(28, 28);
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                const grey = 255 * (1 - array[i][j]);
                imageData.data[(i * 28 + j) * 4] = grey;
                imageData.data[(i * 28 + j) * 4 + 1] = grey;
                imageData.data[(i * 28 + j) * 4 + 2] = grey;
                imageData.data[(i * 28 + j) * 4 + 3] = 255;
            }
        }
        contextDecoded.putImageData(imageData, 0, 0);
    }
    
    function drawLatentAndDecodedTensor(tensor) {
        drawLatentTensor(tensor);
        drawDecodedTensor(decoder.predict(tensor));
    }

    function chooseDigit(differentFrom) {
        if (differentFrom == undefined) {
            return Math.floor(Math.random() * 10);
        }
        let candidate = Math.floor(Math.random() * 9);
        if (candidate >= differentFrom) {
            candidate++;
        }
        return candidate;
    }

    const animationStepDuration = 5000; // ms
    var animationStartMilliseconds = 0;
    var animationFrom = chooseDigit();
    var animationTo = chooseDigit(animationFrom);
    var continueAnimation = false;

    function animate(timeStampMilliseconds) {

        // Update progress
        if (timeStampMilliseconds == undefined) {
            timeStampMilliseconds = 0;
        }
        if (animationStartMilliseconds == undefined) {
            animationStartMilliseconds = timeStampMilliseconds;
        }
        let progress = Math.min(1, Math.max(0, (timeStampMilliseconds - animationStartMilliseconds) / animationStepDuration));
        progress = 1 - 0.5 * (Math.sin(Math.PI * progress + Math.PI / 2) + 1); // ease-in-out
        
        // Compute latent tensor
        const latentTensor = latentBase[animationFrom].mul(1 - progress).add(latentBase[animationTo].mul(progress));
        
        // Display latent tensor
        const latentArray = latentTensor.reshape([16]).arraySync();
        for (let i = 0; i < latentArray.length; i++) {
            latentInputs[i].value = latentArray[i];
        }
        drawLatentAndDecodedTensor(latentTensor);

        // Restart animation
        if (progress == 1) {
            animationFrom = animationTo;
            animationTo = chooseDigit(animationFrom);
            animationStartMilliseconds = undefined;
        }
        if (continueAnimation) {
            requestAnimationFrame(animate);
        }
    }

    const latentInputs = [];
    function createLatentInputs() {
        const container = document.getElementById("latent-inputs");
        container.innerHTML = "";
        for (let p = 0; p < 16; p++) {
            const inputContainer = document.createElement("div");
            inputContainer.classList.add("latent-input-container");
            const input = document.createElement("input");
            input.type = "range";
            input.min = Math.floor(minValues[p]);
            input.max = Math.ceil(maxValues[p]);
            input.value = (minValues[p] + maxValues[p]) / 2;
            input.step = 0.01;
            input.addEventListener("input", updateManualDisplay);
            latentInputs.push(input);
            inputContainer.appendChild(input);
            container.appendChild(inputContainer);
        }
    }
    
    function updateManualDisplay() {
        stopAnimation();
        let values = [];
        for (let i = 0; i < 16; i++) {
            values.push(parseFloat(latentInputs[i].value));
        }
        let tensor = tf.tensor(values).reshape([1, 2, 2, 4]);
        drawLatentAndDecodedTensor(tensor);
    }

    function loadDigit(digit) {
        const array = latentBase[digit].reshape([16]).arraySync();
        for (let i = 0; i < 16; i++) {
            latentInputs[i].value = array[i];
        }
        updateManualDisplay();
    }

    createLatentInputs();

    function createDigitButtons() {
        const container = document.getElementById("digit-buttons");
        container.innerHTML = "";
        for (let digit = 0; digit <= 9; digit++) {
            const button = document.createElement("button");
            button.textContent = digit;
            button.addEventListener("click", () => {
                loadDigit(digit);
            });
            container.appendChild(button);
        }
    }

    createDigitButtons();

    function startAnimation() {
        continueAnimation = true;
        document.getElementById("button-animate").textContent = `Stop`;
        animate();
    }

    function stopAnimation() {
        continueAnimation = false;
        document.getElementById("button-animate").textContent = `Animer`;
    }

    function toggleAnimation() {
        if (continueAnimation) {
            stopAnimation();
        } else {
            startAnimation();
        }
    }

    document.getElementById("button-animate").addEventListener("click", toggleAnimation);

    loadDigit(3);

}

window.addEventListener("load", onWindowLoad);