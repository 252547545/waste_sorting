const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const img2x = (imgPath) => {
  const buffer = fs.readFileSync(imgPath);
  return tf.tidy(() => {
    const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
    const imgTsResized = tf.image.resizeBilinear(imgTs, [224, 224]);
    return imgTsResized.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3]);
  });
}

const getData = async (trainDir, outputDir) => {
  const classes = fs.readdirSync(trainDir).filter(n => !n.includes('.'));
  fs.writeFileSync(`${outputDir}/classes.json`, JSON.stringify(classes));
  //声明类别标签
  const data = [];
  //图片读取
  classes.forEach((dir, dirIndex) => {
    fs.readdirSync(`${trainDir}/${dir}`).filter(n => n.match(/jpg$/)).forEach(filename => {
      const imgPath = `${trainDir}/${dir}/${filename}`;
      data.push({
        imgPath,
        dirIndex
      });
    });
  });
  //洗牌 混着训练
  tf.util.shuffle(data);

  const ds = tf.data.generator(function* () {
    const count = data.length
    const batchSize = 32;
    for (let start = 0; start < count; start += batchSize) {
      const end = Math.min(start + batchSize, count);
      yield tf.tidy(() => {
        const inputs = [];
        const labels = [];
        for (let j = start; j < end; j++) {
          const {
            imgPath,
            dirIndex
          } = data[j];
          const x = img2x(imgPath);
          inputs.push(x);
          labels.push(dirIndex);
        }
        const xs = tf.concat(inputs);
        const ys = tf.tensor(labels);
        return {
          xs,
          ys
        }
      });
    }
  });

  return {
    ds,
    classes
  }
}

module.exports = getData;