
import * as tf from '@tensorflow/tfjs'

export function file2img(f) {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.readAsDataURL(f);
    reader.onload = (e) => {
      const img = document.createElement('img');
      img.src = e.target.result;
      img.width = 224;
      img.height = 224;
      img.onload = () => resolve(img);
    }
  })
}

export function img2x(imgEl) {
  return tf.tidy(() => {
    return tf.browser.fromPixels(imgEl).toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3]);
  })
}