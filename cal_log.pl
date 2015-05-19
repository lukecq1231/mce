#!/usr/bin/perl
$vecfile="vectors.txt";
$setfile="./data/testset.txt"; #testset or devset
$logfile="log_testset.txt";

open VEC,'<',$vecfile or die $!;
open SET,'<',$setfile or die $!;
open LOG,'>',$logfile or die $!;

print "Read GRE set...\n";
while(<SET>){
	chomp;
	my @arr=$_=~/(\S+): (\S+) (\S+) (\S+) (\S+) (\S+) :: (\S+)/;
	# print "@arr\n";
	foreach (@arr){
		$hash{$_}=1;
	}
}
close SET;

print "Read word embedding...\n";
$a = <VEC>;
($vocab,$dimen)=$a=~/(\d+) (\d+)/;
print "word num: $vocab\nvec dimen: $dimen\n";
while(<VEC>){
	chomp;
	my @arr=split /\s+/,$_;
	my $word=shift @arr;
	if(exists $hash{$word}){
		$vecof{$word}="@arr";
	}
}
close VEC;

print "Find antonym...\n";
open SET,'<',$setfile or die "Can not open SET!";
while(<SET>){
	$total++;
	chomp;
	print LOG "$_\n";
	($ques,$c1,$c2,$c3,$c4,$c5,$ans)=$_=~/(\S+): (\S+) (\S+) (\S+) (\S+) (\S+) :: (\S+)/;
	if (!defined($vecof{$ques}) or !&notzero($vecof{$ques})){
		print LOG "$ques is OOV\n\n";
		next;
	}
	my @cs=();
	foreach (($c1,$c2,$c3,$c4,$c5)){
		if (defined($vecof{$_}) and &notzero($vecof{$_})){
			$nowdis=&cosdis($vecof{$ques},$vecof{$_});
			print LOG "$ques $_ -> $nowdis\n";
			push @cs,$_;
		}
		else{
			print LOG "$_ is OOV\n";
		}
	}
	if (@cs){
		$attempt++;
		my $first=shift @cs;
		$mindis=&cosdis($vecof{$ques},$vecof{$first});
		$predict=$first;
		foreach my $nowword (@cs){
			$nowdis=&cosdis($vecof{$ques},$vecof{$nowword});
			if($nowdis<$mindis){
				$predict=$nowword;
				$mindis=$nowdis;
			}
		}
		print LOG "Select answer: $predict\n\n";
		$correct++ if $ans=~/$predict/;
	}
	else{
		print LOG "\n";
	}
}

print "\nnumber of questions answered correctly: $correct\n";
print "number of questions attempted: $attempt\n";
print "total number of questions : $total\n";
$p=$correct/$attempt;
$r=$correct/$total;
$f1=2*$p*$r/($p+$r);
printf "precision : %.2f\n",$p;
printf "recall : %.2f\n",$r;
printf "F-score : %.2f\n",$f1;

close SET;
sub cosdis{
	my($m,$n);
	($m,$n)=@_;
	my @a1=split ' ',$m;
	my @a2=split ' ',$n;
	my $norm1=0;
	my $norm2=0;
	my $sum=0;
	foreach (0..$#a1){
		$norm1+=$a1[$_]*$a1[$_];
		$norm2+=$a2[$_]*$a2[$_];
		$sum+=$a1[$_]*$a2[$_];
	}
	$sum=$sum/(sqrt($norm1)*sqrt($norm2));
}
sub notzero{
	my ($m);
	($m)=@_;
	my @a=split ' ',$m;
	my $norm=0;
	foreach (0..$#a){
		$norm+=$a[$_]*$a[$_];
	}
	if($norm == 0){
		0;
	}
	else{
		1;
	}
}



