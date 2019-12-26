
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../images/Mfcc.png',width=6, height=4,to='(0,2.2,0)' ),

    to_Dropout( "dp0", dp_rate="Dropout 82\%", offset="(-0.2,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_Dense("dense1", 40 ,offset="(2,0,0)", caption="FC CReLU",Color ="\FcCreluColor",depth=50 ),#relu color is default
    to_connection("dp0","dense1"),
   
    #to_SoftMax("soft2", 10 ,"(5,0,0)", caption="SoftMax"  ),
    to_Dense("dense2", 572 ,offset="(4,0,0)", caption="FC Linear",Color ="\FcLinearColor",depth=30 ),
    to_connection("dense1","dense2"),
    
    to_output( '../images/outpu-a.png',width=2, height=4,to='(4,1.1,-2)' ),
    #to_connection("dense1", "soft1"),  
    #to_connection("pool2", "soft1"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
