import { useState, useEffect } from 'react';
import { fonts } from '../data/fontData';

const HandwritingCanvas = ({ text, speed, strokeWidth, color }) => {
    const [paths, setPaths] = useState([]);
    const [totalWidth, setTotalWidth] = useState(0);

    useEffect(() => {
        let currentX = 10;
        const CHAR_WIDTH = 50;
        const SPACING = 5;
        const selectedFont = fonts.clean;

        const newPaths = text.split('').map((char, index) => {
            const d = selectedFont[char] || selectedFont['?'];

            if (char === ' ') {
                currentX += CHAR_WIDTH / 2;
                return null;
            }

            if (!d) {
                // Skip unknown characters for now
                return null;
            }

            const pathData = {
                d,
                x: currentX,
                y: 10,
                id: `char-${index}-${char}`,
                delay: index * speed * 0.5 // Sequential delay
            };

            currentX += CHAR_WIDTH + SPACING;
            return pathData;
        }).filter(Boolean);

        setPaths(newPaths);
        setTotalWidth(currentX + 50);
    }, [text, speed]);

    return (
        <div className="w-full overflow-x-auto p-4 bg-white rounded-lg shadow-inner">
            <svg
                width={totalWidth}
                height={120}
                viewBox={`0 0 ${totalWidth} 120`}
                className="mx-auto"
            >
                {paths.map((path) => (
                    <path
                        key={path.id}
                        d={path.d}
                        transform={`translate(${path.x}, ${path.y})`}
                        fill="none"
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="handwriting-path"
                        style={{
                            '--path-speed': `${speed}s`,
                            '--path-delay': `${path.delay}s`,
                        }}
                    />
                ))}
            </svg>
            <style>{`
            .handwriting-path {
              stroke-dasharray: 300;
              stroke-dashoffset: 300;
              animation: draw var(--path-speed) linear forwards;
              animation-delay: var(--path-delay);
            }
            @keyframes draw {
              to {
                stroke-dashoffset: 0;
              }
            }
          `}</style>
        </div>
    );
};

export default HandwritingCanvas;
