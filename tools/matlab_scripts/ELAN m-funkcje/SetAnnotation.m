function oElan = SetAnnotation(p_oElan, p_sTierName, p_iAnnotationIndex, p_sFields)

%Ustawia pola adnotacji na pozycji p_iAnnotationIndex z warstwy o 
%nazwie p_sTierName obiektu p_oElan. Wartoœci pól okreœla parametr 
%p_sFields.
%
%parametry:
%
%p_oElan - obiekt, na którego danych ma operowaæ funkcja. Parametr 
%ten musi byæ obiektem struktury oElan. W przeciwnym wypadku funkcja
%zostaje przerwana z b³êdem.
%
%p_sTierName - nazwa warstwy, z której adnotacja ma zostaæ 
%zmodyfikowana. Warstwa o danej nazwie musi istnieæ. Parametr ten musi
%byæ typu string. W przeciwnym wypadku funkcja zostaje przerwana 
%z b³êdem.
%
%p_iAnnotationIndex - indeks okreœlaj¹cy pozycjê adnotacji, która ma 
%zostaæ zmodyfikowana. Parametr ten musi przyjmowaæ wartoœæ niewiêksz¹
%ni¿ liczba wszystkich adnotacji w danej warstwie. Musi byæ równie¿ 
%typu numerycznego. W przeciwnym wypadku funkcja zostaje przerwana 
%z b³êdem.
%
%p_sFields - tablica komórek typu string przechowuj¹ca nowe wartoœci 
%pól, które maj¹ zostaæ ustawione w adnotacji. Parametr ten musi byæ 
%typu cell array of strings. W przeciwnym wypadku funkcja zostaje 
%przerwana z b³êdem.
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
	
	if isnumeric(p_iAnnotationIndex) == 0 
		oElan = p_oElan;
		disp('Error: p_iAnnotationIndex is not a number.');
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
			
			if p_iAnnotationIndex > numel(p_oElan.oTiers(i).oAnnotations) || p_iAnnotationIndex < 1
				oElan = p_oElan;
				disp('Error: Annotation of such index does not exist.');
				return;
			end
			
			p_oElan.oTiers(i).oAnnotations(p_iAnnotationIndex).sFields = p_sFields;
			
			bFound = true;
		end
	end
	
	if ~bFound
		oElan = p_oElan;
		disp('Error: Tier of such name does not exist.');
		return;
	end
	
	oElan = p_oElan;
	
end