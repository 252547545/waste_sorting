import React, { PureComponent } from "react";
import { file2img, img2x } from './utils'
import { Button, Progress } from 'antd'
import 'antd/dist/antd.css'
import * as tf from '@tensorflow/tfjs'
import intro from './intro'

const DATA_URL = 'http://127.0.0.1:8080'

class app extends PureComponent {
  state = {}

  async componentDidMount() {
    this.model = await tf.loadLayersModel(DATA_URL + '/model.json');
    //this.model.summary();
    this.classes = await fetch(DATA_URL + '/classes.json').then(res => res.json());
  }
  predict = async (file) => {
    const img = await file2img(file);
    this.setState({ imgSrc: img.src });
    setTimeout(() => {
      const pred = tf.tidy(() => {
        const x = img2x(img);
        return this.model.predict(x);
      });
      const results = pred.arraySync()[0].map((score, i) => ({ score, label: this.classes[i] })).sort((a, b) => b.score - a.score)
      this.setState({ results });
    }, 0);
  }
  renderResult = (item) => {
    const finalScore = Math.round(item.score * 100);
    return (
      <tr key={item.label}>
        <td>{item.label}</td>
        <td><Progress percent={finalScore} status={finalScore == 100 ? 'success' : 'normal'} /></td>
      </tr>
    )
  }
  render() {
    const { imgSrc, results } = this.state;
    const finalItem = results && { ...results[0], ...intro[results[0].label] };
    return (
      <div style={{ padding: 20 }} >
        <Button type="primary" size="large" style={{ width: '100%' }}
          onClick={() => { this.upload.click() }} >点击上传</Button>
        <input type="file" onChange={e => this.predict(e.target.files[0])}
          ref={el => { this.upload = el }} style={{ display: 'none' }} />
        {imgSrc && <div style={{ marginTop: 20, textAlign: "center" }}>
          <img src={imgSrc} style={{ maxWidth: '100%', height: 300 }} />
        </div>}
        {finalItem && <div>识别结果:</div>}
        {finalItem && <div style={{ display: 'flex', alignItems: 'flex-start', marginTop: 20 }}>
          <img src={finalItem.icon} width={120} />
          <div>
            <h2 style={{ color: finalItem.color }}>{finalItem.label}</h2>
            <div style={{ color: finalItem.color }}>{finalItem.intro}</div>
          </div>
        </div>}
        {results && <div style={{ marginTop: 20 }} >
          <table style={{ width: '100%' }}>
            <tbody>
              <tr><td style={{ width: 80, padding: '5px 0' }}>类别</td><td>匹配度</td></tr>
              {results.map(this.renderResult)}
            </tbody>
          </table>
        </div>

        }
      </div>
    )
  }
}

export default app;