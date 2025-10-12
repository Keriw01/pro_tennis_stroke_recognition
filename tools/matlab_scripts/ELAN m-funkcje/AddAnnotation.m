function oElan = AddAnnotation(p_oElan, p_sTierName, p_iStartFrame, p_iStopFrame, p_sFields)

%Dodaje adnotacjê do warstwy o nazwie p_sTierName obiektu p_oElan.
%Zamkniêty przedzia³ czasowy adnotacji okreœlony jest przez parametry 
%p_iStartFrame ip_iStopFrame jako kolejno pocz¹tek i koniec. Adnotacja
%posiada pola okreœlone przez parametr p_sFields.
%
%parametry:
%
%p_oElan - obiekt, na którego danych ma operowaæ funkcja. Parametr ten
%musi byæ obiektem struktury oElan. W przeciwnym wypadku funkcja 
%zostaje przerwana z b³êdem.
%
%p_sTierName - nazwa warstwy, do której ma zostaæ dodana adnotacja. 
%Warstwa o danej nazwie musi istnieæ. Parametr ten musi byæ typu 
%string. W przeciwnym wypadku funkcja zostaje przerwana z b³êdem.
%
%p_iStartFrame - pocz¹tek przedzia³u zamkniêtego adnotacji wyra¿ony 
%w ramkach. Parametr ten musi przyjmowaæ wartoœæ wiêksz¹ ni¿ 0 i 
%mniejsz¹ ni¿ p_iStopFrame. Musi byæ równie¿ typu numerycznego. W 
%przeciwnym wypadku funkcja zostaje przerwana z b³êdem.
%
%p_iStopFrame - koniec przedzia³u zamkniêtego adnotacji wyra¿ony w 
%ramkach. Parametr ten musi przyjmowaæ wartoœæ wiêksz¹ ni¿ 0 i wiêksz¹
%ni¿ p_iStartFrame. Musi byæ równie¿ typu numerycznego. W przeciwnym 
%wypadku funkcja zostaje przerwana z b³êdem.
%
%p_sFields - tablica komórek typu string przechowuj¹ca pola adnotacji.
%Parametr ten musi byæ typu cell array of strings. W przeciwnym 
%wypadku funkcja zostaje przerwana z b³êdem.
%
%wartoœæ zwracana:
%
%Zaktualizowany obiekt struktury oElan.

	if isstruct(p_oElan) == 0
		oElan = [];	
		disp('Error: p_oElan is not a structure');
		return;
	end
	
	if ~isfield(p_oElan, 'oTiers') || ~isfield(p_oElan.oTiers, 'oAnnotations') || ~isfield(p_oElan.oTiers, 'sName')
		oElan = [];
		disp('Error: p_oElan is a wrong structure');
		return;
	end
	
	if ischar(p_sTierName) == 0 
		oElan = p_oElan;
		disp('Error: p_sTierName is not a string.');
		return;
	end
	
	if isnumeric(p_iStartFrame) == 0 
		oElan = p_oElan;
		disp('Error: p_iStartFrame is not a number.');
		return;
	end
	
	if p_iStartFrame < 0 
		oElan = p_oElan;
		disp('Error: p_iStartFrame is below zero.');
		return;
	end
	
	if isnumeric(p_iStopFrame) == 0 
		oElan = p_oElan;
		disp('Error: p_iStopFrame is not a number.');
		return;
	end
	
	if p_iStopFrame < 0 
		oElan = p_oElan;
		disp('Error: p_iStopFrame is below zero.');
		return;
	end
	
	if p_iStartFrame > p_iStopFrame
		oElan = p_oElan;
		disp('Error: p_iStartFrame is greater than p_iStopFrame.');
		return;
	end
	
	if iscellstr(p_sFields) == 0 
		oElan = p_oElan;
		disp('Error: p_sFields is not a cell array of strings.');
		return;
	end
	
	bFound = false;
	for i = 1 : numel(p_oElan.oTiers)
		sTemp = p_oElan.oTiers(i).sName;
		if strcmp(p_sTierName, sTemp)	
		
			if numel(p_oElan.oTiers(i).oAnnotations) == 0
				p_oElan.oTiers(i).oAnnotations(1).iStartIndex = p_iStartFrame;
				p_oElan.oTiers(i).oAnnotations(1).iStopIndex = p_iStopFrame;
				p_oElan.oTiers(i).oAnnotations(1).sFields = p_sFields;
                p_oElan.oTiers(i).oAnnotations(1).bIsReference = 0;
                p_oElan.oTiers(i).oAnnotations(1).sRefTierName = [];
				oElan = p_oElan;
				return
			end
			
			for j = 1 : numel(p_oElan.oTiers(i).oAnnotations)
						
				if p_iStartFrame < p_oElan.oTiers(i).oAnnotations(j).iStartIndex
					if j ~= 1
						if p_iStartFrame < p_oElan.oTiers(i).oAnnotations(j-1).iStopIndex
							oElan = p_oElan;
							disp('Error: Wrong annotaion range.');
							return;
						end
					end
					if p_iStopFrame > p_oElan.oTiers(i).oAnnotations(j).iStartIndex
						oElan = p_oElan;
						disp('Error: Wrong annotaion range.');
						return;
					end
					%insert annotation
					oTempAnnotations = p_oElan.oTiers(i).oAnnotations;
					
					for k = j+1 : numel(oTempAnnotations)+1
						p_oElan.oTiers(i).oAnnotations(k) = oTempAnnotations(k-1);
					end
					
					p_oElan.oTiers(i).oAnnotations(j).iStartIndex = p_iStartFrame;
					p_oElan.oTiers(i).oAnnotations(j).iStopIndex = p_iStopFrame;
					p_oElan.oTiers(i).oAnnotations(j).sFields = p_sFields;
                    p_oElan.oTiers(i).oAnnotations(j).bIsReference = 0;
                    p_oElan.oTiers(i).oAnnotations(j).sRefTierName = [];
					oElan = p_oElan;
					return
					%insert annotation end
				end				
			end %for j		
							
			if p_iStartFrame < p_oElan.oTiers(i).oAnnotations(j).iStopIndex
				oElan = p_oElan;
				disp('Error: Wrong annotaion range.');
				return;
			end
			
			p_oElan.oTiers(i).oAnnotations(end+1).iStartIndex = p_iStartFrame;
			p_oElan.oTiers(i).oAnnotations(end).iStopIndex = p_iStopFrame;
			p_oElan.oTiers(i).oAnnotations(end).sFields = p_sFields;
            p_oElan.oTiers(i).oAnnotations(end).bIsReference = 0;
            p_oElan.oTiers(i).oAnnotations(end).sRefTierName = [];
			oElan = p_oElan;
			return;		
			
			bFound = true;
		end 
	end %for i
	
	if ~bFound
		oElan = p_oElan;
		disp('Error: Tier of such name does not exist.');
		return;
	end	
end