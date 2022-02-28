#Script for jack-knife

getFullTable <- function(path) {
	name_txt <- data.frame()
	z<-data.frame()
	temp = list.files(path = path, pattern="*.breakage")   
	for (i in 1:length(temp)) {
		a<-read.table(paste0(path,temp[i],sep=""), sep="\t", row.names=1, stringsAsFactors=F)
		if (length(name_txt) == 0) {
			name_txt <- rownames(a)
		} else {
			name_txt <- intersect(name_txt, rownames(a))
		}
	}
	for (i in 1:length(temp)) {
		a<-read.table(paste0(path,temp[i],sep=""), sep="\t", row.names=1, stringsAsFactors=F)
		if (length(z) == 0) {
			z <- a[name_txt,]
		} else {
			z<-cbind(z, a[name_txt,])
		}
	}
	rownames(z) <- name_txt
	z1<-z[apply(z, 1, function(x) { return(sum(x < 0.2) != length(x))}),]
	return(z1)
}

find_max_var <- function(x) {
	max_diff <- -9999999999999.9
	diff_index <- -1
	for (i in 2:length(x)) {
		diff <- x[i]-x[i-1]
		if (max_diff < diff) {
			max_diff <- diff
			diff_index <- i
		}
	}
	return(diff_index)
}
#First Argument is directory with control breakages
#Second Argument is directory with tumor breakages
args <- commandArgs()
con <- getFullTable(args[6])
exp <- getFullTable(args[7])
intintint <- intersect(rownames(con), rownames(exp))   
library(diptest)
library(modes)
library(uniftest)
full_table <- cbind(con[intintint,], exp[intintint,])  
#For script speed up we use only significant differ CpG islands with only two modes
ff<-unlist(lapply(intintint, function(x) {
	len <- length(full_table[x,])
	nn <- find_max_var(sort(full_table[x,]))
	if ((nn > 2)&&(nn < len-2)) {
		return(wilcox.test(full_table[x,1:nn], full_table[x,(nn+1):len])$p.value < 0.07)
	}
	else {
		return(FALSE)
	}
}))

full_table <- full_table[ff,]
tp <- 0
fp <- 0
tn <- 0
fn <- 0
res_or<-c(rep.int(0, length(colnames(con))), rep.int(1, length(colnames(exp))))
library(e1071)
counter <- 0
pos <- 0
neg <- 0
while (counter < 500) {
	print(1:length(colnames(con)))
	i <- sample(1:length(colnames(con)),1,TRUE)
	j <- sample(length(colnames(con)):length(res_or)-1,1,TRUE)
	res_ss <- res_or[-c(i,j)]
	a<-svm(res_ss ~ ., t(full_table[,-c(i,j)]), kernel="linear")
#	print(i)
	res <- predict(a, t(full_table[,i]))
#	print(res)
#	print(res_or[i])
	if (res_or[i] == 1) {
		pos <- pos + 1
	}
	if (res_or[i] == 0) {
		neg <- neg + 1
	}

	if ((res < 0.5)&&(res_or[i] == 0)) {
		tn <- tn + 1
	}
	if ((res < 0.5)&&(res_or[i] == 1)) {
		fn <- fn + 1
	}
	if ((res > 0.5)&&(res_or[i] == 1)) {
		tp <- tp + 1
	}
	if ((res > 0.5)&&(res_or[i] == 0)) {
		fp <- fp + 1
	}
	res <- predict(a, t(full_table[,j]))
	if (res_or[j] == 1) {
		pos <- pos + 1
	}
	if (res_or[j] == 0) {
		neg <- neg + 1
	}

	if ((res < 0.5)&&(res_or[j] == 0)) {
		tn <- tn + 1
	}
	if ((res < 0.5)&&(res_or[j] == 1)) {
		fn <- fn + 1
	}
	if ((res > 0.5)&&(res_or[j] == 1)) {
		tp <- tp + 1
	}
	if ((res > 0.5)&&(res_or[j] == 0)) {
		fp <- fp + 1
	}
	counter <- counter+2
}
#write.table(file='file.txt', x=full_table)
print("________________")
print(tp)
print(fp)
print(tn)
print(fn)
print(pos)
print(neg)
