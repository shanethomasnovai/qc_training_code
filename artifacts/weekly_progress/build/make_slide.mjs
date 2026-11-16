import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/Shane/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const require=createRequire(runtime+'/node/node_modules/preview.cjs');
const {Presentation,PresentationFile}=require('@oai/artifact-tool');
const skill='C:/Users/Shane/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
const workspace=process.cwd();
const build=path.join(workspace,'artifacts/weekly_progress/build');
const output=path.join(workspace,'artifacts/weekly_progress/output');
await fs.mkdir(output,{recursive:true});
const d=JSON.parse(await fs.readFile(path.join(build,'data.json'),'utf8'));
const pres=Presentation.create({slideSize:{width:1600,height:900}});
const slide=pres.slides.add();
slide.background.fill='#FFFFFF';
function text(name,content,x,y,w,h,size=24,bold=false,color='#171717',underline=false){
  const s=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=content;
  s.text.style={fontSize:size,typeface:'Arial',bold,color,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,...(underline?{underline:'sng'}:{})};
  return s;
}
function rule(name,x,y,w,h=2){slide.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill:'#222222',line:{fill:'none',width:0}});}
text('Title','Patch ResNet reaches κ = 0.791 and 84.6% binary F1',64,27,1475,60,44,true,'#171717',true);
rule('Title divider',2,94,1596);
text('Architecture','ImageNet-pretrained ResNet-18 with ordinal and binary heads (λ = 1).',64,113,1460,34,26);
text('Population','Same patient split: 508 training and 127 validation images. Single seed 42; no test evaluation.',64,155,1470,33,25);
rule('Body divider',2,214,1596);
rule('Column divider',1007,216,2,638);
text('Results heading','Validation performance (seed 42)',64,248,915,39,28,true);
const percent=v=>(v*100).toFixed(1)+'%';
const labels=['IQAnet, patches','ResNet-18, patches','ResNet-18, patches','ResNet-18, whole image','ResNet-18, whole image'];
const lrs=['0.0001','0.0001','0.00001','0.00001','0.00003'];
const values=[['Training method','LR','κ','Binary F1','Sensitivity','Specificity'],...d.models.map((r,i)=>[labels[i],lrs[i],r.qwk.toFixed(3),percent(r.f1),percent(r.sensitivity),percent(r.specificity)])];
const table=slide.tables.add({rows:6,columns:6,left:64,top:300,width:914,height:306,columnWidths:[333,111,80,126,132,132],values});
table.borders.assign({fill:'#222222',width:1.2,style:'solid'});
for(let r=0;r<6;r++){
  table.rows[r].height=51;
  for(let c=0;c<6;c++){
    const cell=table.getCell(r,c);
    cell.fill=r===0?'#595959':r===3?'#E5E5E5':'#FFFFFF';
    cell.text.style={fontSize:r===0?22:23,typeface:'Arial',bold:r===0||r===3,color:r===0?'#FFFFFF':'#171717',alignment:c===0?'left':'center',verticalAlignment:'middle',insets:{left:14,right:9,top:8,bottom:8},wrap:'none'};
  }
}
text('Results interpretation','Lower LR improved patch ResNet κ from 0.762 to 0.791.\nWhole images trained about 9× faster per epoch,\nbut patches gave better grading results in these runs.',64,635,913,104,25);
text('Next experiment','Next: repeat the best patch model with seeds 7 and 123',64,760,914,38,25,true);
text('Workflow progress','Added a universal config and automatic Excel experiment logging.',64,808,914,36,23);

text('Distribution heading','Grade mix: both graders combined',1042,248,510,37,27,true);
text('Distribution population','Training + validation: 217 baseline, 418 non-baseline',1042,289,510,51,21);
const chartRows=d.groups.map(group=>{
  const counts=[0,1,2,3,4].map(k=>group.graders[0].counts[k]+group.graders[1].counts[k]);
  if(counts.reduce((a,b)=>a+b,0)!==group.n*2)throw new Error('Rating counts do not reconcile');
  return {category:group.group,r:{counts,percent:counts.map(x=>x/(group.n*2))}};
});
const colors=['#E1E1E1','#B7B7B7','#8C8C8C','#555555','#222222'];
const chart=slide.charts.add('bar',{
  name:'Grade distributions',
  position:{left:1025,top:352,width:540,height:265},
  categories:chartRows.map(r=>r.category),
  series:[0,1,2,3,4].map(k=>({name:'Grade '+(k+1),values:chartRows.map(x=>Number(x.r.percent[k].toPrecision(12))),fill:colors[k],
    line:{fill:'#FFFFFF',width:1},valuesFormatCode:'0.0%',
    dataLabels:{showValue:true,position:'center',numberFormatCode:'0%',textStyle:{fontSize:18,fill:k<2?'#171717':'#FFFFFF'}},
    dataLabelOverrides:chartRows.map((x,idx)=>({idx,showValue:x.r.percent[k]>=.095}))})),
  barOptions:{direction:'bar',grouping:'stacked',gapWidth:65,overlap:100},
  hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17,fill:'#171717'}},
  xAxis:{min:0,max:1,majorUnit:.25,numberFormatCode:'0%',textStyle:{fontSize:17,fill:'#444444'},majorGridlines:{fill:'#DDDDDD',width:.7},line:{fill:'#999999',width:.7}},
  yAxis:{min:0,max:1,majorUnit:.25,textStyle:{fontSize:19,fill:'#171717'},majorGridlines:null,line:{fill:'none',width:0}},
  chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},
});
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
applyPresentationChartFont(chart,{fontFamily:'Arial'});
for(const series of chart.series.items)series.dataLabels.textStyle.typeface='Arial';
text('Grade 5 values','Grade 5: 2.1% baseline, 0.7% non-baseline',1042,637,507,35,20);
text('Distribution interpretation','Combined grades 1–2:\n48.8% baseline vs 44.1% non-baseline\nGrade 1: 15.4% vs 11.1%',1042,691,507,98,23);
text('Distribution definition','Shane + Ganaav ratings pooled: two votes per image.\n434 baseline ratings and 836 non-baseline ratings.\nBaseline follows the “.0” filename rule.',1042,801,510,70,18);

slide.speakerNotes.text=[
 'Weekly progress for the data science meeting on 12 November 2026. New experiments were completed 6–11 November. Reference design supplied by Shane as a screenshot; rebuilt as editable text, a table and a native chart.',
 'All displayed model results use the same 127 validation images and seed 42. These are single runs, not three-seed means. IQAnet and ResNet both use joint ordinal/binary heads with lambda=1. ResNet starts from torchvision ResNet18_Weights.IMAGENET1K_V1, then fine-tunes all layers. Architecture and pretraining change together in the IQAnet comparison.',
 'Kappa is quadratic-weighted kappa against the saved consensus reference. Binary F1, sensitivity and specificity come from the dedicated binary head at threshold 0.5. Grades 1–2 are acceptable. Tied references use the existing evaluation conventions. Single-seed differences are exploratory.',
 'Patch input: resize to 768x768, nine 256x256 grayscale crops, average reconstructed grade probabilities. Whole-image input: resize once to 256x256, no crop. Both repeat grayscale to RGB and apply ImageNet normalization. Same seed, optimizer, dropout, loss and patient split. LR3e-5 run retains the same ReduceLROnPlateau settings as LR1e-5, so its rate can decrease to the existing 1e-5 floor.',
 'Training/validation label population: 635 images, excluding the held-out VISUS-NHP test set. Baseline operational definition: literal .0 anywhere in image_id, consistent with the existing baseline split script. It is not a separate verification of clinical visit timing. Group sizes are 217 baseline and 418 non-baseline. The chart pools the two integer ratings from Shane and Ganaav for every image: 434 baseline rating votes and 836 non-baseline rating votes. Ratings are not averaged or rounded. Grade charts use current reviewed ratings, never the ignored grading column in the uploaded attribute CSV. Two votes from an image are not independent observations.',
 ...chartRows.map(x=>`${x.category} combined Shane + Ganaav: counts ${x.r.counts.join(', ')}; exact percentages ${x.r.percent.map(p=>(100*p).toFixed(4)).join(', ')}.`),
 ...d.models.map(r=>r.label+'; metrics source: '+r.path+'/validation_metrics.json; QWK '+r.qwk+'; binary F1 '+r.f1),
 ...d.groups.flatMap(group=>group.graders.map(g=>`${group.group}; ${g.grader}; ${g.rated}/${group.n} rated; counts grades 1–5: ${g.counts.join(', ')}.`)),
 'Label source: C:/Users/Shane/Desktop/qc_data/datasets/reviewed_v2/multigrader_labels.xlsx and its verified saved patient split. Additional reviewers each rated 11/217 baseline and 25/418 non-baseline images. The combined distribution includes only Shane and Ganaav, as requested.',
 'Whole images used about one minute per epoch versus 8–9 minutes for patch ResNet on the same CPU. Whole images have nine times fewer training inputs and optimizer steps per epoch; epochs therefore are not equal amounts of training. This experiment changes context and image detail together. It does not establish an inherent superiority of patches after independent tuning.',
 'Exploratory attribute analysis also examined blur/noise labels. Reviewer disagreement and tied quality grades prevented a reliable attribute-specific weakness from being established; no attribute augmentation was introduced. Experiment logging is automatic after successful runs. Next proposed validation is additional seeds, keeping the held-out test set untouched.',
].join('\n\n');
await fs.writeFile(path.join(build,'preview.png'),new Uint8Array(await (await pres.export({slide,format:'png',scale:1})).arrayBuffer()));
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const finalPath=path.join(output,'weekly_progress_12_nov_2026.pptx');
const result=await finalizePresentation({
 workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','1'],
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[1],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json'),
});
console.log(JSON.stringify({finalPath,result}));
