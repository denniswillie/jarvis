import React from 'react';
import {Composition} from 'remotion';
import {Clip} from './Clip.mjs';
import {COMPOSITION_ID, FPS, HEIGHT, WIDTH, validateClipProps} from './props.mjs';

export function Root() {
  return React.createElement(Composition, {
    id: COMPOSITION_ID,
    component: Clip,
    width: WIDTH, height: HEIGHT, fps: FPS,
    calculateMetadata: ({props}) => {
      const checked = validateClipProps(props);
      return {durationInFrames: checked.recipe.durationInFrames, props: checked};
    },
  });
}
