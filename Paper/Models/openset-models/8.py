
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../images/espectrogram.png',width=6, height=1.7,to='(0,1,-0.2)' ),

    to_Dropout( "dp0", dp_rate="Dropout 82\%", offset="(-0.2,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=""),
    to_Rnn("Rnn1", 128 ,offset="(2,0,0)", caption="RNN ReLU",Color ="\FcReluColor",depth=5,width=5, height=5 ),#relu color is default
    to_connection("dp0","Rnn1"),
    #to_SoftMax("soft1", 10 ,"(2,0,0)", caption="Elu"  ),
    
    to_Rnn("Rnn2", 80 ,offset="(5,0,0)", caption="RNN LeakyReLU ",Color ="\FcEluColor",depth=5,width=5, height=5 ),#relu color is default
    to_connection("Rnn1","Rnn2"),

    to_Dense("dense1", 40 ,offset="(8,0,0)", caption="FC LeakyReLU ",Color ="\FcEluColor",depth=30 ),
    to_connection("Rnn2","dense1"),


    to_Dense("dense3", 572 ,offset="(10,0,0)", caption="FC Linear",Color ="\FcLinearColor",depth=30 ),
    to_connection("dense1","dense3"),

    to_output( '../images/outpu-a.png',width=2, height=4,to='(10,1.0,-1)' ),
    
    #to_connection("dense1", "soft1"),  
    #to_connection("pool2", "soft1"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
