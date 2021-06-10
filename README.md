# BA-Thesis RNN

Repozitorij za završni rad  
Tema: "Implementacija povratne neuronske mreže i primjena na zadatku predviđanja sljedeće riječi"  
Student: Ivan Martinović  
Mentor: izv. prof. dr. sc Jan Šnajder  
Komentor: mag. ing. Josip Jukić  

Prilikom vježbanja i učenja strukture mreža korišteni su primjeri sa sljedećih poveznica:  
    - https://peterroelants.github.io/posts/neural-network-implementation-part01/  
    - https://peterroelants.github.io/posts/rnn-implementation-part01/  
    - https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/  

  
Lokacija implementiranih slojeva: /source-py, unutar tog direktorija nalaze se: DenseLayer, RNNLayer, LSTMLayer, GRULayer itd.  
Napomena: Backward metoda GRULayer-a trenutno nije potpuno funkcionalna (neispravni gradijenti s obzirom na skriveno stanje).  

Dataset: https://arxiv.org/abs/1710.03957  

