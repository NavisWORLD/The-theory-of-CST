const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

export function projectCinematic({dx=0,dy=0,width=1280,height=720,zoom=1}){
  const anchorX=width*0.32;
  const anchorY=height*0.72;
  const depth=clamp((-dy+450)/1350,0,1);
  const scale=clamp(1.16-depth*.58,.5,1.22)*zoom;
  return {x:anchorX+dx*.86*zoom,y:anchorY+dy*.34*zoom,scale,depth};
}

export function atmosphereProfile(environment={}){
  const solar=clamp(environment.solar??.55,0,1);
  const dust=clamp(environment.dust??.15,0,1);
  const severity=environment.storm?.type==='dust'?clamp(environment.storm.severity??.6,0,1):0;
  return {
    daylight:.18+solar*.82,
    haze:clamp(.12+dust*.42+severity*.4,0,1),
    dustOpacity:clamp(.04+dust*.18+severity*.34,0,.68),
    sunStrength:clamp(solar*(1-dust*.42)*(1-severity*.35),.05,1),
    stormSeverity:severity
  };
}
