% FUNCTION ShortTimeEnergy : windowLength and step in # of samples

function E = ShortTimeEnergy(signal, windowLength,step)

curPos = 1;
L = length(signal);
E = [];
frameCounter=1;
while (curPos+windowLength-1<=L)    
    window = (signal(curPos:curPos+windowLength-1));
    E(frameCounter) = (1/(windowLength)) * sum(abs(window.^2));
    curPos = curPos + step;
    frameCounter=frameCounter+1;
end
