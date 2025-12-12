import { useState } from 'react'
import './App.css'
import HandwritingCanvas from './components/HandwritingCanvas'

function App() {
  const [text, setText] = useState('Hello World')
  const [speed, setSpeed] = useState(0.5)
  const [strokeWidth, setStrokeWidth] = useState(3)
  const [color, setColor] = useState('#333333')

  return (
    <div className="app-container">
      <h1>SVG Handwriting Path</h1>
      <p className="subtitle">Sans IA - Pure SVG Animation</p>

      <div className="main-content">
        <div className="controls-panel">
          <div className="control-group">
            <label>Text to Write</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows="3"
              placeholder="Type something..."
            />
          </div>

          <div className="settings-grid">
            <div className="control-group">
              <label>Speed ({speed}s)</label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
              />
            </div>

            <div className="control-group">
              <label>Thickness ({strokeWidth}px)</label>
              <input
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={strokeWidth}
                onChange={(e) => setStrokeWidth(parseFloat(e.target.value))}
              />
            </div>

            <div className="control-group">
              <label>Color</label>
              <div className="color-picker-wrapper">
                <input
                  type="color"
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                />
                <span>{color}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="preview-area">
          <h2>Preview</h2>
          <HandwritingCanvas
            text={text}
            speed={speed}
            strokeWidth={strokeWidth}
            color={color}
          />
          <button className="replay-btn" onClick={() => {
            // Force re-render hack to replay animation
            const currentText = text;
            setText('');
            setTimeout(() => setText(currentText), 10);
          }}>
            Replay Animation
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
