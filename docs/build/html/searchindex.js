Search.setIndex({docnames:["functions","functions.astro","functions.comp","functions.cosmo2","functions.errors","functions.extra_math","functions.fits","functions.image","functions.interface","functions.log","functions.matrix","functions.np_adjust","functions.shape","functions.signal","functions.stats","functions.string","functions.system","functions.types","index","lib","lib.algorithms","lib.convolve","lib.cost","lib.deconvolve","lib.directional","lib.file_io","lib.gradient","lib.linear","lib.noise","lib.optimisation","lib.other_methods","lib.plotting","lib.proximity","lib.quality","lib.reweight","lib.sf_deconvolve_args","lib.shape","lib.svd","lib.tests","lib.transform","lib.wavelet","modules","sf_deconvolve"],envversion:50,filenames:["functions.rst","functions.astro.rst","functions.comp.rst","functions.cosmo2.rst","functions.errors.rst","functions.extra_math.rst","functions.fits.rst","functions.image.rst","functions.interface.rst","functions.log.rst","functions.matrix.rst","functions.np_adjust.rst","functions.shape.rst","functions.signal.rst","functions.stats.rst","functions.string.rst","functions.system.rst","functions.types.rst","index.rst","lib.rst","lib.algorithms.rst","lib.convolve.rst","lib.cost.rst","lib.deconvolve.rst","lib.directional.rst","lib.file_io.rst","lib.gradient.rst","lib.linear.rst","lib.noise.rst","lib.optimisation.rst","lib.other_methods.rst","lib.plotting.rst","lib.proximity.rst","lib.quality.rst","lib.reweight.rst","lib.sf_deconvolve_args.rst","lib.shape.rst","lib.svd.rst","lib.tests.rst","lib.transform.rst","lib.wavelet.rst","modules.rst","sf_deconvolve.rst"],objects:{"":{lib:[19,0,0,"-"],sf_deconvolve:[42,0,0,"-"]},"lib.algorithms":{PowerMethod:[20,1,1,""]},"lib.algorithms.PowerMethod":{get_spec_rad:[20,2,1,""],set_initial_x:[20,2,1,""]},"lib.cost":{sf_deconvolveCost:[22,1,1,""]},"lib.cost.sf_deconvolveCost":{calc_cost:[22,2,1,""],grad_comp:[22,2,1,""],lowr_comp:[22,2,1,""],psf_comp:[22,2,1,""],sparse_comp:[22,2,1,""]},"lib.deconvolve":{get_lambda:[23,3,1,""],perform_reweighting:[23,3,1,""],run:[23,3,1,""],set_condat_param:[23,3,1,""],set_grad_op:[23,3,1,""],set_linear_op:[23,3,1,""],set_lowr_thresh:[23,3,1,""],set_noise:[23,3,1,""],set_optimisation:[23,3,1,""],set_primal_dual:[23,3,1,""],set_prox_op_and_cost:[23,3,1,""],set_sparse_weights:[23,3,1,""]},"lib.directional":{convolve_dir_filters:[24,3,1,""],get_dir_filters:[24,3,1,""]},"lib.file_io":{check_data_format:[25,3,1,""],read_file:[25,3,1,""],read_from_fits:[25,3,1,""],read_input_files:[25,3,1,""],write_output_files:[25,3,1,""],write_to_fits:[25,3,1,""]},"lib.gradient":{GradKnownPSF:[26,1,1,""],GradNone:[26,1,1,""],GradPSF:[26,1,1,""],GradUnknownPSF:[26,1,1,""]},"lib.gradient.GradKnownPSF":{get_grad:[26,2,1,""]},"lib.gradient.GradNone":{get_grad:[26,2,1,""]},"lib.gradient.GradPSF":{H_op:[26,2,1,""],Ht_op:[26,2,1,""]},"lib.gradient.GradUnknownPSF":{get_grad:[26,2,1,""]},"lib.noise":{add_noise:[28,3,1,""],denoise:[28,3,1,""]},"lib.optimisation":{Condat:[29,1,1,""],FISTA:[29,1,1,""],ForwardBackward:[29,1,1,""],GenForwardBackward:[29,1,1,""]},"lib.optimisation.Condat":{iterate:[29,2,1,""],update:[29,2,1,""],update_param:[29,2,1,""]},"lib.optimisation.FISTA":{speed_switch:[29,2,1,""],speed_up:[29,2,1,""],update_lambda:[29,2,1,""]},"lib.optimisation.ForwardBackward":{iterate:[29,2,1,""],update:[29,2,1,""]},"lib.optimisation.GenForwardBackward":{iterate:[29,2,1,""],update:[29,2,1,""]},"lib.plotting":{plotCost:[31,3,1,""]},"lib.quality":{e_error:[33,3,1,""],nmse:[33,3,1,""]},"lib.reweight":{cwbReweight:[34,1,1,""]},"lib.reweight.cwbReweight":{reweight:[34,2,1,""]},"lib.sf_deconvolve_args":{ArgParser:[35,1,1,""],get_opts:[35,3,1,""]},"lib.sf_deconvolve_args.ArgParser":{convert_arg_line_to_args:[35,2,1,""]},"lib.shape":{Ellipticity:[36,1,1,""],ellipticity_atoms:[36,3,1,""],shape_project:[36,3,1,""]},"lib.shape.Ellipticity":{get_centroid:[36,2,1,""],get_ellipse:[36,2,1,""],get_moments:[36,2,1,""],update_centroid:[36,2,1,""],update_weights:[36,2,1,""],update_xy:[36,2,1,""]},"lib.tests":{test_deconvolution:[38,3,1,""],test_images:[38,3,1,""],test_psf_estimation:[38,3,1,""]},lib:{algorithms:[20,0,0,"-"],cost:[22,0,0,"-"],deconvolve:[23,0,0,"-"],directional:[24,0,0,"-"],file_io:[25,0,0,"-"],gradient:[26,0,0,"-"],noise:[28,0,0,"-"],optimisation:[29,0,0,"-"],plotting:[31,0,0,"-"],quality:[33,0,0,"-"],reweight:[34,0,0,"-"],sf_deconvolve_args:[35,0,0,"-"],shape:[36,0,0,"-"],tests:[38,0,0,"-"]},sf_deconvolve:{check_psf:[42,3,1,""],main:[42,3,1,""],run_script:[42,3,1,""],set_out_string:[42,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"2007n":36,"3103c":36,"abstract":36,"case":26,"class":[20,22,23,26,29,33,34,35,36],"default":[20,22,28,29,33,35,36,38],"final":31,"float":[20,22,23,24,26,28,29,33,34],"function":[18,22,23,29,31,33,41],"import":[33,36],"int":[20,23,24,25,29,36,38],"ngol\u00e9":[33,36],"return":[22,23,24,25,26,28,29,33,35,36,38],"short":18,"true":[20,22,29,38],For:[25,33],The:[23,26,36],abs:36,absolut:23,action:26,activ:29,add:[23,28],add_nois:28,added:28,adding:28,addition:23,after:22,algorithm:[18,19,23,29,41],algoritm:29,allow:35,alreadi:23,analys:[33,36],analysi:34,angl:24,angle_num:24,aplli:33,appendix:36,applic:[29,34],approxim:18,arg:[22,35],argpars:35,argument:[23,35],argumentpars:35,arrai:[20,22,23,24,25,26,28,34,36,38,42],arrrai:20,arxiv:36,assess:33,astro:[0,41],author:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],auto_iter:29,auto_run:20,automat:[20,29],avail:18,averag:38,b2010:29,backward:29,baker:36,base:[20,22,24,26,29,33,34,35,36],basic:20,bauschk:29,been:42,begin:29,beta_reg:26,between:22,binari:25,blank:35,blanklin:36,bm2007:36,bool:[20,22,29,36],boyd:34,c2013:[29,36],cacluat:36,calc_cost:22,calcualt:20,calcul:[20,22,23,26,36],call:29,cand:34,centroid:36,cfm:36,chapter:29,check:[23,25,42],check_data_format:25,check_psf:42,classs:26,clean:[23,38],clean_data_fil:38,code:[18,42],coeffici:[22,24],com:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],combin:23,comment:35,comp:[0,41],compon:[22,36],composit:29,comput:33,condat:[23,29],constraint:29,contain:[19,20,24,26,28,29,31,33,34,36,38,42],content:41,contraint:22,control:26,converg:[20,23,29],convert:35,convert_arg_line_to_arg:35,convex:29,convolut:26,convolv:[19,24,41],convolve_dir_filt:24,correct:25,cosmo2:[0,41],cost:[19,23,29,31,41],cost_list:31,creat:42,criteria:29,cropper:36,current:[25,29,36],current_file_nam:25,custom:35,cwb2007:34,cwbreweight:34,data:[20,22,23,24,25,26,28,33,34,36,38,42],data_file_nam:25,data_shap:[20,23],date:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],deafult:36,decconvolv:23,deconv_data:38,deconvolut:[22,23,25,26,35,38],deconvolution_script:[25,35],deconvolv:[18,19,22,38,41,42],defin:[20,23,25,26,35,36],defualt:[26,36],defult:[22,35],denois:28,describ:[18,34],develop:36,deviat:[23,28,38],dict:23,differ:22,dimens:[25,28,33,36],direct:[19,41],distanc:[33,36],doe:[25,28,38],domain:23,dual:[23,25,29],dual_r:25,dummi:26,e_error:33,either:29,ellipt:[33,36,38],ellipticity_atom:36,engin:29,enhanc:34,equat:[29,33,36,42],error:[0,22,33,36,38,41],estim:[22,36,38],exampl:[33,36],execut:42,expect:25,experi:36,exponenti:36,express:36,extens:25,extra_math:[0,41],f2017:42,factor:34,fals:36,farren:[18,19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],field:[33,36],file:[25,31,38],file_io:[19,41],file_nam:25,filter:[23,24],first:[28,33],fista:29,fit:[0,25,41],fix:[26,29],follow:[33,36],form:22,format:[25,35],forward:29,forwardbackward:29,fourier:34,fred:[24,33,36],from:[25,26,28,29,33,34,35,36,42],front:36,fulli:26,galaxi:[23,36,42],gauss:28,gaussian:[28,38],gener:[29,36],genforwardbackward:29,get:[20,22,23,24,26,35],get_centroid:36,get_dir_filt:24,get_ellips:36,get_grad:26,get_lambda:23,get_moment:36,get_opt:35,get_spec_rad:20,given:[26,36,38],gmail:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],grad:[22,29],grad_comp:22,grad_typ:22,gradient:[19,22,23,29,41],gradknownpsf:26,gradnon:26,gradpsf:26,gradunknownpsf:26,guess:29,h_op:26,hao:24,hard:28,hartman:36,has:[25,42],here:18,ht_op:26,http:36,imag:[0,18,23,25,26,33,36,38,41,42],image1:33,image2:33,implement:[18,19,20,22,29,33,34,36,42],inhereit:29,inherit:26,ini:35,initi:[20,22,23,29],initialis:[20,23,29,36],input:[20,22,23,24,25,26,28,33,35,36,42],instanc:[20,23,42],interfac:[0,41],intermedi:29,invalid:[25,33],invers:29,involv:29,iter:[20,26,29,36],its:36,journal:[29,34],kernel:38,keyword:23,known:[23,26],kwagr:23,kwarg:[23,35],l1norm:23,lambda:[23,29],lambda_init:29,lambda_lowr:22,lambda_psf:22,lambda_reg:26,lambda_upd:29,lead:23,learn:[33,36],lens:36,less:25,level:[23,28],lib:[18,41],librari:19,line:35,linear:[19,23,29,41],lipschitzian:29,list:[25,28,29,31,36],log:[0,41,42],logger:42,low:[18,22,23],lowr:22,lowr_comp:22,mad:23,main:42,make:31,match:[25,28,38],math:26,matrix:[0,22,26,36,41],max_it:[20,29],maximum:[20,29],mean:[33,38],measur:38,median:[23,38],met:29,method:[20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],metric:[33,38],minim:34,mnra:36,moallem:36,mode:[22,42],modul:[18,41],moment:36,n_dim:25,n_imag:23,n_iter:36,name:[25,31,38],namespac:[35,42],ndarrai:[22,23,24,25,26,28,29,33,34,36,38,42],ngole:[24,33,36],nmse:33,nois:[19,23,26,41],noise_typ:28,noisi:[23,25],none:[22,25,29,31,36,38],norm:[22,23],normalis:[33,36,42],note:[26,29,33,34,36,42],np_adjust:[0,41],npy:25,ns2016:[33,36],nuber:24,nuclear:22,number:[20,23,25,28,29,36,38],numpi:[25,36],obj_var:26,object:[20,22,23,26,29,34],observ:[26,42],obtain:22,off:29,offset:36,onli:[22,28],oper:[20,22,23,26,29],opt:42,optic:36,optim:[29,33,36],optimis:[19,22,23,34,41],option:[20,22,25,28,29,31,35,36,38],opttimis:23,org:36,origin:[22,33],osapublish:36,other_method:[19,41],ouput:42,output:[22,25,31,42],output_file_nam:25,output_format:25,overrid:35,p_pixel:23,packag:[18,41],paper:18,paramat:29,paramet:[20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],parser:35,part:36,path:25,perfom:28,perform:[20,23],perform_reweight:23,pixel:[23,38],place:29,plot:[19,41],plotcost:31,point:29,poisson:28,posit:[22,29,36],power:20,powermethod:[20,26],primal:[23,25,29],primal_r:25,problem:[23,29],project:36,properli:42,properti:26,provid:[23,24,29],prox:[26,29],prox_dual:29,prox_list:29,proxim:[19,23,26,29,41],psf:[22,23,25,26,33,35,36,38,42],psf_comp:22,psf_data:38,psf_file_nam:25,psf_re:25,psf_type:26,psf_unknown:22,psnr:38,q00:36,q01:36,q10:36,q11:36,quadrupol:36,qualiti:[19,38,41],r2012:29,radiu:[20,23],raguet:29,rais:[25,28,33,38],random:[20,38],random_se:38,rank:[18,22,23],reach:29,read:25,read_fil:25,read_from_fit:25,read_input_fil:25,reconstruct:[29,33],recov:[26,38],refer:[29,33,34,36],regular:22,regularis:[18,23,26],relax:29,remov:28,result:[25,26,33,38],retunr:25,return_norm:36,reweight:[19,23,41],rho:29,rho_upd:29,rotat:[24,26],routin:[24,28,31,33,36],run:[23,42],run_script:42,samuel:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],scheme:34,scienc:29,script:[35,42],sec:33,second:33,section:34,seed:38,sensor:36,set:[20,23,29,35,42],set_condat_param:23,set_grad_op:23,set_initial_x:20,set_linear_op:23,set_lowr_thresh:23,set_nois:23,set_optimis:23,set_out_str:42,set_primal_du:23,set_prox_op_and_cost:23,set_sparse_weight:23,sf_deconvolv:19,sf_deconvolve_arg:[19,41],sf_deconvolvecost:22,sf_deonvolv:22,sf_tool:26,shack:36,shan:24,shape:[0,19,20,23,24,25,41],shape_project:36,siam:[33,36],sigma:[23,24,28,29,34,36],sigma_upd:29,signal:[0,41],singl:[23,26],singular:23,size:[26,36],skip:35,soft:28,some:[33,36],sourc:[22,23,24,25,26,38,42],space:36,spars:[22,23],sparse_comp:22,sparsiti:[18,22,23,34],spec_rad:23,specif:26,specifi:[25,42],spectral:[20,23],speed:29,speed_switch:29,speed_up:29,split:[23,29],squar:33,stack:[33,38],standard:[23,28,29,38],starck:[33,36],stat:[0,41],step:[26,29],str:[22,25,26,28,31,35,38],string:[0,35,41,42],style:35,submodul:41,summari:18,svd:[19,41],swicth:29,system:[0,41],tau:[23,29],tau_upd:29,techniqu:[23,36],term:29,test:[19,22,41],test_deconvolut:38,test_imag:38,test_psf_estim:38,than:25,them:23,theori:29,thi:[18,19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],thresh_factor:34,threshold:[20,22,23,28,34],threshold_typ:28,toler:20,total:23,transform:[19,24,41],transport:[33,36],transpos:26,true_psf_fil:38,truth:38,tupl:[20,23,24,25,36],turn:29,turn_on:29,two:33,type:[0,22,23,24,25,26,28,33,35,36,38,41,42],unknown:26,until:29,updat:[23,26,29,36,42],update_centroid:36,update_lambda:29,update_param:29,update_weight:36,update_xi:36,upon:[20,29],uri:36,use:[23,29],use_fista:29,used:[23,29],using:[18,23,36],valu:[20,23,26,28,29,31,33,36],valueerror:[25,28,33,38],variabl:[23,29],variant:[23,26],verbos:22,version:[19,20,22,23,24,25,26,28,29,31,33,34,35,36,38,42],wakin:34,wave:36,wavelet:[19,22,23,41],wavelet_filt:23,weak:36,weight:[22,23,29,34,36],when:26,width:24,work:[24,33,36],write:25,write_output_fil:25,write_to_fit:25,www:36,x_new:29,x_old:29,x_prox:29,x_temp:29,yield:35,zero:[26,36]},titles:["functions package","functions.astro module","functions.comp module","functions.cosmo2 module","functions.errors module","functions.extra_math module","functions.fits module","functions.image module","functions.interface module","functions.log module","functions.matrix module","functions.np_adjust module","functions.shape module","functions.signal module","functions.stats module","functions.string module","functions.system module","functions.types module","SF_DECONVOLVE Documentation","lib package","lib.algorithms module","lib.convolve module","lib.cost module","lib.deconvolve module","lib.directional module","lib.file_io module","lib.gradient module","lib.linear module","lib.noise module","lib.optimisation module","lib.other_methods module","lib.plotting module","lib.proximity module","lib.quality module","lib.reweight module","lib.sf_deconvolve_args module","lib.shape module","lib.svd module","lib.tests module","lib.transform module","lib.wavelet module","sf_deconvolve","sf_deconvolve module"],titleterms:{"function":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],algorithm:20,astro:1,comp:2,content:[0,18,19],convolv:21,cosmo2:3,cost:22,deconvolv:23,direct:24,document:18,error:4,extra_math:5,file_io:25,fit:6,gradient:26,imag:7,interfac:8,lib:[19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],linear:27,log:9,matrix:10,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42],nois:28,np_adjust:11,optimis:29,other_method:30,packag:[0,19],plot:31,proxim:32,qualiti:33,reweight:34,sf_deconvolv:[18,41,42],sf_deconvolve_arg:35,shape:[12,36],signal:13,stat:14,string:15,submodul:[0,19],svd:37,system:16,test:38,todo:23,transform:39,type:17,wavelet:40}})