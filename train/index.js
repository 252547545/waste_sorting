const tf = require('@tensorflow/tfjs-node');
const getData = require('./data');

const TRAIN_DIR = '垃圾分类/train';
const OUTPUT_DIR = 'output';
//mobilenet网络地址
const MOBILENET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json';

const main = async () => {

  //加载数据
  const {
    ds,
    classes
  } = await getData(TRAIN_DIR, OUTPUT_DIR);

  //定义模型
  const mobilenet = await tf.loadLayersModel(MOBILENET_URL);
  mobilenet.summary();
  const model = tf.sequential();
  //模型输出
  //console.log(mobilenet.layers.map((l, i) => [l.name, i]));
  //mobilenet有1000多种类别我们复用了前86种类别
  for (let i = 0; i <= 86; i++) {
    const layers = mobilenet.layers[i];
    layers.trainable = false;
    model.add(layers);
  }
  //将高维网络摊平
  model.add(tf.layers.flatten());
  //定义双层神经网络做分类任务
  model.add(tf.layers.dense({
    units: 10, //神经网络设置10
    activation: 'relu',
  }));
  //多分类激活函数
  model.add(tf.layers.dense({
    units: classes.length,
    activation: 'softmax'
  }))

  //训练模型
  model.compile({
    loss: 'sparseCategoricalCrossentropy',
    optimizer: tf.train.adam(),
    metrics: ['acc']
  })
  await model.fitDataset(ds, {
    epochs: 20
  });
  await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`);
}

main();