from scipy.stats import binom

nr_pasi=16
probabilitate=1/2
#poz finala=nr_pasi_inainte-nr_pasi_inapoi
#nr_pasi_inainte+nr_pasi_inapoi=nr_pasi
#nr_pasi_inapoi=nr_pasi-nr_pasi_inainte
#poz finala=nr_pasi_inainte-(nr_pasi-nr_pasi_inainte)=2*nr_pasi_inainte-nr_pasi
#8=2*nr_pasi_inainte-16
#24=2*nr_pasi_inainte
#12=nr_pasi_inainte
pasi_inainte=12
#P(pasi_inainte=12)=nr_pasi!/((n-pasi_inainte)!*pasi_inainte!)*p_succes^12*(1-p_succes)^(nr_pasi-nr_pasi_inainte)
#P(pasi_inainte=12)=16!/(4!*12!) * (1/2)^12 * (1/2)^4
#P(pasi_inainte=12)=13*14*15*16/(2*3*4)*(1/2)^16
#P(pasi_inainte=12)=13*7*5*4*(1/2^16)
#P(pasi_inainte=12)=13*7*5/2^14
#P(pasi_inainte=12)=455/16384=0,0277709
print(binom.pmf(pasi_inainte,nr_pasi,probabilitate))