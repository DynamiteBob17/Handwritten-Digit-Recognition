import './App.scss';
import {visor, show, metrics, render} from '@tensorflow/tfjs-vis';
import {MnistData} from './mnist/data.js';
import {useEffect, useRef, useState} from 'react';
import * as tf from '@tensorflow/tfjs';
import {ReactSketchCanvas} from 'react-sketch-canvas';
import {Button} from '@mui/material';

function App() {
    const [data, setData] = useState(null);
    const [trainingModel, setTrainingModel] = useState(false);
    const [modelTrained, setModelTrained] = useState(false);
    const [predictions, setPredictions] = useState(new Array(10).fill(0));
    const canvas = useRef(null);
    const model = useRef(null);
    const CLASS_NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

    function getModel() {
        const sequentialModel = tf.sequential();

        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;
        const IMAGE_CHANNELS = 1;

        sequentialModel.add(tf.layers.conv2d({
            inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        sequentialModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

        sequentialModel.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        sequentialModel.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

        sequentialModel.add(tf.layers.flatten());

        const NUM_OUTPUT_CLASSES = 10;
        sequentialModel.add(tf.layers.dense({
            units: NUM_OUTPUT_CLASSES,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));

        const optimizer = tf.train.adam();
        sequentialModel.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return sequentialModel;
    }

    async function createModel() {
        const newModel = getModel();
        model.current = newModel;
    }

    useEffect(() => {
        (async () => {
            const mnistData = new MnistData();
            await mnistData.load();
            setData(mnistData);

            await createModel();
        })();
    }, []);

    async function train(data) {
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const container = {
            name: 'Model Training', tab: 'Model', styles: {height: '1000px'}
        };
        const fitCallbacks = show.fitCallbacks(container, metrics);

        const BATCH_SIZE = 512;
        const TRAIN_DATA_SIZE = 5500;
        const TEST_DATA_SIZE = 1000;

        const [trainXs, trainYs] = tf.tidy(() => {
            const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
            return [
                d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        });

        const [testXs, testYs] = tf.tidy(() => {
            const d = data.nextTestBatch(TEST_DATA_SIZE);
            return [
                d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        })

        return model.current.fit(trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            validationData: [testXs, testYs],
            epochs: 10,
            shuffle: true,
            callbacks: fitCallbacks
        });
    }

    function doPrediction(data, testDataSize = 500) {
        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;
        const testData = data.nextTestBatch(testDataSize);
        const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
        const labels = testData.labels.argMax(-1);
        const preds = model.current.predict(testxs).argMax(-1);

        testxs.dispose();
        return [preds, labels];
    }

    async function showAccuracy(data) {
        const [preds, labels] = doPrediction(data);
        const classAccuracy = await metrics.perClassAccuracy(labels, preds);
        const container = {name: 'Accuracy', tab: 'Evaluation'};
        await show.perClassAccuracy(container, classAccuracy, CLASS_NAMES);

        labels.dispose();
    }

    async function showConfusion(data) {
        const [preds, labels] = doPrediction(data);
        const confusionMatrix = await metrics.confusionMatrix(labels, preds);
        const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
        await render.confusionMatrix(container, {values: confusionMatrix, tickLabels: CLASS_NAMES});

        labels.dispose();
    }

    async function predictDrawing() {
        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;

        const image = new Image();
        image.src = await canvas.current.exportImage();

        await new Promise(resolve => {
            image.onload = resolve;
        })

        const resizedImage = tf.tidy(() => {
            const tensor = tf.browser.fromPixels(image, 1);
            return tf.image.resizeBilinear(tensor, [IMAGE_WIDTH, IMAGE_HEIGHT]);
        });

        const inputTensor = tf.tidy(() => {
            const normalizedTensor = resizedImage.toFloat().div(tf.scalar(255));
            const reshapedTensor = normalizedTensor.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
            return reshapedTensor;
        });

        const predictions = model.current.predict(inputTensor);
        const predictedLabels = predictions.dataSync();

        resizedImage.dispose();
        inputTensor.dispose();
        predictions.dispose();

        setPredictions(Array.from(predictedLabels));
    }

    async function trainModel() {
        await train(data);
        await showAccuracy(data);
        await showConfusion(data);
    }

    async function renderTensorFlowUIAndTrainModel() {
        if (data) {
            setTrainingModel(true);

            const surface = visor().surface({
                name: 'Input Data Examples', tab: 'Input Data'
            });

            const examples = data.nextTestBatch(20);
            const numExamples = examples.xs.shape[0];

            for (let i = 0; i < numExamples; ++i) {
                const imageTensor = tf.tidy(() => {
                    return examples.xs
                        .slice([i, 0], [1, examples.xs.shape[1]])
                        .reshape([28, 28, 1]);
                });

                const canvas = document.createElement('canvas');
                canvas.width = 28;
                canvas.height = 28;
                canvas.style = 'margin: 4px;';
                await tf.browser.toPixels(imageTensor, canvas);
                surface.drawArea.appendChild(canvas);
                imageTensor.dispose();
            }

            await show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model.current);

            if (!modelTrained) {
                await trainModel();
                setTrainingModel(false);
                setModelTrained(true);
            }
        }
    }

    let trainButtonText;
    if (trainingModel) {
        trainButtonText = 'Training in progress...';
    } else if (!data) {
        trainButtonText = 'Fetching MNIST data...';
    } else {
        trainButtonText = 'Train NN';
    }
    const trainButton = <Button
        onClick={renderTensorFlowUIAndTrainModel}
        disabled={!data || modelTrained || trainingModel}
        size={'small'}
        variant={'contained'}
    >
        {trainButtonText}
    </Button>

    return (
        <>
            <h1 style={{fontStyle: 'italic'}}>Handwritten Digit Recognition</h1>
            <h3 style={{fontStyle: 'italic'}}>by a Neural Network</h3>
            <div className={'App'}>
                <div id={'sketch_area'}>
                    {modelTrained ? <div style={{color: 'green'}}>MODEL TRAINED!</div> : trainButton}
                    <div id={'sketch_canvas_wrapper'}>
                        <ReactSketchCanvas
                            width={'200px'}
                            height={'200px'}
                            strokeWidth={14}
                            strokeColor={'white'}
                            canvasColor={'black'}
                            ref={canvas}
                        />
                    </div>
                    <Button
                        onClick={async () => await predictDrawing()}
                        disabled={trainingModel}
                        size={'small'}
                        variant={'contained'}
                        color={'success'}
                    >
                        Predict
                    </Button>
                    <Button
                        onClick={() => {
                            canvas.current.clearCanvas();
                            setPredictions(new Array(10).fill(0));
                        }}
                        size={'small'}
                        variant={'contained'}
                        color={'error'}
                    >CLEAR
                    </Button>
                </div>
                <div>
                    {
                        predictions.map((pred, idx) => {
                            return <div key={`digit${idx}`} className={'digit_prediction'}>
                                <div
                                    className={'digit_box'}
                                    style={{backgroundColor: `rgba(66,146,198, ${pred})`}}
                                >
                                    {idx}
                                </div>
                                <div style={{fontStyle: 'italic'}}>: {`${(pred * 100).toFixed(3)} %`}</div>
                            </div>
                        })
                    }
                </div>
            </div>
        </>
    );
}

export default App;
