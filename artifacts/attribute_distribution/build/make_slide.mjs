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
const build=path.join(workspace,'artifacts/attribute_distribution/build');
const output=path.join(workspace,'artifacts/attribute_distribution/output');
await fs.mkdir(output,{recursive:true});
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
text('Title','Reviewers differ in how they label image defects',64,27,1475,60,44,true,true);
rule('Title divider',2,94,1596);
text('Scope','Shane and Ganaav annotations on the same 635 training and validation images.',64,113,1470,35,26);
text('Definition','Each bar shows the percentage of images marked TRUE for that attribute. Held-out test images excluded.',64,157,1470,35,24);
rule('Body divider',2,214,1596);
rule('Column divider',1104,216,2,593);
text('Chart heading','Attribute frequency by grader',64,248,1010,37,28,true);
const order=d.attributes.map((_,i)=>i).reverse();
const chart=slide.charts.add('bar',{
 position:{left:45,top:296,width:1030,height:510},
 categories:order.map(i=>d.attributes[i]),
 series:d.graders.map((g,i)=>({name:g.name,values:order.map(k=>Number(g.proportions[k].toPrecision(12))),
  fill:i===0?'#404040':'#B9B9B9',line:{fill:'none',width:0},valuesFormatCode:'0.0%',
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',numberFormatSourceLinked:false,textStyle:{fontSize:21,fill:'#171717'}}})),
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:55,overlap:0},
 hasLegend:true,legend:{position:'top',textStyle:{fontSize:23,fill:'#171717'}},
 xAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:{fontSize:23,fill:'#171717'},line:{fill:'#999999',width:.8},majorGridlines:null},
 yAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:{fontSize:20,fill:'#444444'},line:{fill:'#999999',width:.8},majorGridlines:{fill:'#DDDDDD',width:.7}},
 chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},
});
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
applyPresentationChartFont(chart,{fontFamily:'Arial'});
for(const series of chart.series.items)series.dataLabels.textStyle.typeface='Arial';
text('Blur heading','Blur differs most',1140,260,400,40,28,true);
text('Blur observation','Shane: 15.9%\nGanaav: 72.0%\n\nThis suggests differences in how\nthe blur label was applied.',1140,312,400,174,24);
text('Lighting heading','Lighting tags also differ',1140,501,400,37,26,true);
text('Lighting observation','Uneven illumination: 22.0% vs 5.2%\nDark: 15.0% vs 1.6%\n(Shane vs Ganaav)',1140,550,400,99,23);
text('Next step heading','Next step',1140,678,400,37,27,true);
text('Next step','Review shared examples and agree\non tag definitions before combining\nannotations for modelling.',1140,725,400,96,24);
text('Axis definition','Images tagged (%)',505,802,400,31,22);
text('Multi-label note','Images may have multiple tags, so percentages do not sum to 100%. Tag frequency does not measure annotation accuracy.',64,853,1470,29,20);
slide.speakerNotes.text=[
 'Attribute distribution for the data science presentation, 12 November 2026. Design follows the previous reference slide: white background, Arial, underlined takeaway title and horizontal/vertical rules. Native grouped bars remain editable with an embedded workbook.',
 'Population: exactly the 635 saved training and validation images (508 train and 127 validation), with the held-out VISUS-NHP test images excluded. Both reviewers have a recorded TRUE/FALSE value for each displayed attribute for every selected image. Match by full image ID after stripping folder and image extension. No grade columns from the supplied Shane CSV were loaded or used.',
 'Each percentage is the number of selected images tagged TRUE divided by 635. Tags are non-exclusive. Equal tag frequencies would not imply agreement on which images were tagged. The chart measures prevalence of annotations, not accuracy, physical severity, or whether either reviewer is correct. Different definitions or thresholds are a possible explanation of the differences, not a demonstrated causal result.',
 ...d.graders.flatMap(g=>[g.name+' source: '+g.source,...d.attributes.map((a,i)=>`${g.name}; ${a}: ${g.counts[i]}/635 (${(g.proportions[i]*100).toFixed(4)}%).`)]),
 'Population source: C:/Users/Shane/Desktop/qc_data/datasets/reviewed_v2/multigrader_labels.xlsx and its verified saved patient split. Only image identities and split membership are used from that workbook. Shane source: C:/Users/Shane/Downloads/shane-image-file-listing (3).csv. Ganaav source: C:/Users/Shane/Desktop/qc_data/labels/ganaav_labels.csv.',
 'Reject is omitted because both sources marked it FALSE for every image in this population. Bright is retained despite only one positive annotation per reviewer; similar counts do not prove it is the same image. Minor numeric rounding in embedded chart values preserves 12 significant digits and all displayed percentages. No model or training data was changed.',
].join('\n\n');
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const finalPath=path.join(output,'image_quality_attribute_distribution.pptx');
await finalizePresentation({
 workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[1],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation.json'),
});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.writeFile(path.join(build,'preview.png'),new Uint8Array(await (await final.export({slide:final.slides.items[0],format:'png',scale:1})).arrayBuffer()));
console.log(finalPath);
