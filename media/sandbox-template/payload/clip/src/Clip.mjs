import React, {useEffect, useState} from 'react';
import {AbsoluteFill, Sequence, OffthreadVideo, staticFile, delayRender, continueRender, cancelRender} from 'remotion';
import {segmentVideoProps, validateClipProps} from './props.mjs';

export function CaptionText({text}) {
  return React.createElement('div', {
    style: {
      position: 'absolute', bottom: 112, left: 36, right: 36, padding: '20px 24px',
      color: '#ffffff', backgroundColor: '#000000dd', borderRadius: 12,
      fontFamily: 'Noto Sans', fontWeight: 600, fontSize: 32, lineHeight: 1.35,
      textAlign: 'center', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere',
    },
  }, text); // Literal text child: no HTML, Markdown, template or URL interpretation.
}

export function Clip(input) {
  const {recipe, sources} = validateClipProps(input);
  const [fontHandle] = useState(() => delayRender('Load bundled Noto Sans'));
  const captionText = recipe.captions.map((caption) => caption.text).join(' ') || ' ';
  useEffect(() => {
    let active = true;
    document.fonts.load('600 32px "Noto Sans"', captionText).then((fonts) => {
      if (!active) return;
      if (fonts.length === 0) cancelRender(new Error('Bundled caption font unavailable'));
      else continueRender(fontHandle);
    }, (error) => { if (active) cancelRender(error); });
    return () => { active = false; };
  }, [fontHandle, captionText]);
  // staticFile creates a URL on Remotion's local bundle server. The accepted
  // path is fixed and hash-named; no caller controls its origin or query string.
  const localUrl = staticFile(sources[0].assetPath);
  return React.createElement(AbsoluteFill, {style: {backgroundColor: '#000000'}},
    ...recipe.segments.map((segment) => React.createElement(Sequence, {
      key: segment.segmentId, from: segment.outputFromFrame, durationInFrames: segment.durationInFrames,
    }, React.createElement(OffthreadVideo, segmentVideoProps(segment, recipe.audio, localUrl)))),
    ...recipe.captions.map((caption) => React.createElement(Sequence, {
      key: caption.captionId, from: caption.fromFrame, durationInFrames: caption.toFrame - caption.fromFrame,
    }, React.createElement(CaptionText, {text: caption.text}))),
  );
}
