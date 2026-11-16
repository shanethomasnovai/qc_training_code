import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/Shane/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const require=createRequire(runtime+'/node/node_modules/preview.cjs');
const {Presentation,PresentationFile,FileBlob}=require('@oai/artifact-tool');
const skill='C:/Users/Shane/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
const workspace=process.cwd();
const build=path.join(workspace,'artifacts/weekly_progress/build');
const output=path.join(workspace,'artifacts/weekly_progress/output');
const d=JSON.parse(await fs.readFile(path.join(build,'data.json'),'utf8'));
const pres=Presentation.create({slideSize:{width:1600,height:900}});
const slide=pres.slides.add();
slide.background.fill='#FFFFFF';
function text(name,content,x,y,w,h,size=24,bold=false,underline=false){
 const s=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=content;
 s.text.style={fontSize:size,typeface:'Arial',bold,color:'#171717',verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,...(underline?{underline:'sng'}:{})};
 return s;
}
function rule(name,x,y,w,h=2){slide.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill:'#222222',line:{fill:'none',width:0}});}
text('Title','Patch ResNet reaches κ = 0.791 and 84.6% binary F1',64,27,1475,60,44,true,true);
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
text('Results interpretation','Lower LR improved patch ResNet κ from 0.762 to 0.791.\nGrade accuracy rose from 61.4% for IQAnet to 65.4%.\nWhole images trained about 9× faster per epoch,\nbut gave lower grading performance in these runs.',64,635,913,138,25);
text('Workflow progress','Workflow: one universal config and automatic Excel logging',64,803,914,42,24,true);
text('Next steps heading','Next steps',1042,248,510,42,30,true);
text('Repeatability heading','1. Confirm the patch result',1042,315,510,38,26,true);
text('Repeatability detail','Repeat LR 0.00001 with seeds 7 and 123.\nReport mean κ and binary F1 across\nall three seeds, with their variation.',1042,364,510,108,25);
text('Depth heading','2. Evaluate deeper IQAnet',1042,508,510,38,26,true);
text('Depth detail','4-convolution pilot: κ = 0.770.\nSensitivity 94.7%, specificity 74.3%.\nRepeat across seeds and compare accuracy\nand runtime with the original IQAnet.',1042,557,510,132,25);
text('Evaluation heading','Final evaluation',1042,704,510,38,26,true);
text('Evaluation detail','Keep the patient split fixed during tuning.\nEvaluate the held-out test set after\nselecting the final model.',1042,753,510,108,25);
slide.speakerNotes.text=[
 'Weekly progress for the data science meeting on 12 November 2026. Experiments shown completed 6–11 November 2026. This revision omits the baseline grade-mix analysis at the user’s request.',
 'All results use the same 127 validation images and seed 42. These are individual runs, not means across three seeds. IQAnet and ResNet use joint ordinal/binary heads, lambda=1. ResNet-18 starts from torchvision ImageNet1K V1 weights and fine-tunes all layers. Comparing IQAnet with pretrained ResNet changes both architecture and pretraining.',
 'Kappa is quadratic-weighted kappa against the saved consensus reference. Binary F1, sensitivity and specificity refer to the dedicated binary head with threshold 0.5. Grades 1–2 are acceptable. Existing tie conventions apply. Differences from one seed remain exploratory.',
 'Patch input: resize to 768x768, then nine 256x256 grayscale crops. Average reconstructed grade probabilities over patches. Whole-image input: resize directly to 256x256. Both modes repeat grayscale to RGB, use ImageNet normalization, and retain horizontal flips for training. Whole-image input trades fine detail for global context.',
 'All other settings match between the lower-learning-rate patch run and the first whole-image run. The second whole-image run starts at LR 0.00003 and retains the same scheduler floor of 0.00001. Nine times fewer training inputs produce roughly nine times fewer optimizer steps per whole-image epoch. The speed statement is per epoch, not per equivalent training effort.',
 ...d.models.map(r=>r.label+'; source: '+r.path+'/validation_metrics.json; QWK '+r.qwk+'; accuracy '+r.accuracy+'; dedicated binary F1 '+r.f1+'; sensitivity '+r.sensitivity+'; specificity '+r.specificity),
 'Next steps distinguish completed pilots from planned repeats. Repeat the best patch ResNet configuration with seeds 7 and 123 alongside existing seed 42, then report mean and variation. A four-convolution IQAnet pilot has already completed: QWK 0.7700236801782978, dedicated binary F1 0.8372093023255814, sensitivity 0.9473684210526315, specificity 0.7428571428571429. It detected 54/57 unacceptable images and correctly accepted 52/70 acceptable images. Best checkpoint: epoch 59; early stopping: epoch 69. Source: C:/Users/Shane/Desktop/qc_data/models/iqanet4_or_soft_binary_lambda1_seed42_20261111_155229_571655_22b51ca3/validation_metrics.json. Further evaluation across seeds and runtime comparison are proposed next steps, not completed repeats. The original patch-level loss, max/min pooling and equal patch averaging were retained. Keep held-out test data untouched until final selection.',
 'The universal YAML selects backbone and input mode. Completed local runs automatically update qc_data/experiment_log.xlsx. No baseline grade-distribution chart is included in this revision.',
].join('\n\n');
const candidate=path.join(build,'results_only_candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const finalPath=path.join(output,'weekly_progress_with_iqanet4_next_step.pptx');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({
 workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','1'],
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'iqanet4_next_step_validation.json'),
});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.writeFile(path.join(build,'results_only_preview.png'),new Uint8Array(await (await final.export({slide:final.slides.items[0],format:'png',scale:1})).arrayBuffer()));
console.log(finalPath);
