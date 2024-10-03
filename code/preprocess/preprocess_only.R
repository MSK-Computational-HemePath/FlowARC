library(flowCore)
library(flowDensity)

save_gate_plot <- function(fcs_path){
	x <- read.FCS(fcs_path, transformation=F) 
	fcsname <- tools::file_path_sans_ext(basename(fcs_path))
	fcsdirname <- dirname(fcs_path)
	removeMargins <- function(flowFrame,dimensions){
	  # Look at the accepted ranges for the dimensions
	  meta <- pData(flowFrame@parameters)
	  rownames(meta) <- meta[,"name"]
	  
	  # Initialize variables
	  selection <- rep(TRUE,times=dim(flowFrame)[1])
	  e <- exprs(flowFrame)
	  
	  # Make selection
	  for(d in dimensions){
	    selection <- selection & 
	      e[,d] > max(meta[d,"minRange"],min(e[,d])) &
	      e[,d] < min(meta[d,"maxRange"],max(e[,d]))
	  }
	  return(flowFrame[selection,])
	}
	
	removeDoublets <- function(flowFrame, d1="FSC-A", d2="FSC-H", w=NULL,silent=TRUE){
	  # Calculate the ratios
	  ratio <- exprs(flowFrame)[,d1] / (1+ exprs(flowFrame)[,d2])
	  
	  # Define the region that is accepted
	  r <- median(ratio)
	  if(is.null(w)){ w <- 3.8*sd(ratio) }
	  if(!silent){
	    print(r)
	    print(w)
	  }
	  
	  # Make selection
	  selection <- which(ratio < r+w)
	  return(flowFrame[selection,])
	}
	
	x_pre <- removeMargins(x,dimensions=colnames(x))
	x_rmd <- removeDoublets(x_pre,d1="FSC-A",d2="FSC-H")
	
    # get spill matrix
	comp_list <- spillover(x)
	comp  <- comp_list[[1]]
    #compensate
	x_comp <- compensate(x_rmd, comp)

    #transform cell features appropriately
	linear <- scaleTransform(a=0,b=262144)
	fklog <- function( transformId, s=262144, m=5){
	      t = new("transform", .Data = function(x) ((1/m)*log(x/s,10))+1)
	      t@transformationId = transformId
	      t
	    }
	logT <- fklog(transformId="FK", s=262144,m=5)
	tl <- transformList(c("FSC-A","FSC-H","SSC-A","SSC-H","Time"), list(linear, linear,linear,logT,linear))
	x_pretrans <- transform(x_comp, tl)
	logiclecols <- colnames(x_comp)[5:14]
	logicle <- logicleTransform(w = 0.6, t = 262144, m = 4.5, a = 0)
	tl2 <-  transformList(logiclecols, list(logicle,logicle,logicle,logicle,logicle,logicle,logicle,logicle,logicle))
	x_trans0 <- transform(x_pretrans, tl2)
	linear2 <- scaleTransform(a=0,b=4.5)
	tl3 <-  transformList(logiclecols, list(linear2,linear2,linear2,linear2,linear2,linear2,linear2,linear2,linear2))
	x_trans <- transform(x_trans0, tl3)

    #save final preprocessed fcs file
	write.FCS(x_trans, paste(folderexppth,paste('/',paste(fcsname,'.fcs',sep=""), sep = ""),sep=""), what="numeric", delimiter = "|", endian="big")
	###
	
}

# Path to the tumor gated fcs files of all tumor cases
folderorigpth <- "../original_data/tumor_gate_tumor_cases"
# Path to the preprocessed tumor gated fcs files of all tumor cases
folderexppth <- "../tmp/cases_tumor_cells"

fcslist <- list.files(folderorigpth,full.names=TRUE,pattern="*.fcs")

for (x in fcslist) {
	try(namefcs <- save_gate_plot(x), silent= TRUE)
}