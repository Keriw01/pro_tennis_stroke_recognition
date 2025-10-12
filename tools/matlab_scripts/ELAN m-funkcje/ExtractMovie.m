function ExtractMovie(p_sInputAviFile, p_sOutputAviFile, p_oAnnotation, p_bUseCompression)

%Wycina i zapisuje na dysku twardym fragment filmu o nazwie 
%p_sInputAviFile. Wyciêty fragment odpowiada przedzia³om czasowym 
%adnotacji p_oAnnotation. Jest on zapisywany do pliku o nazwie 
%p_sOutputAviFile. Zapisywany fragment filmu jest kompresowany 
%(kompresja Cinepak) jeœli parametr p_bUseCompression ma wartoœæ true.
%Jeœli wejœciowy film wideo (p_sInputAviFile) jest skompresowany, 
%funkcja ta wymaga aby w systemie operacyjnym zainstalowany by³ kodek,
%który potrafi go zdekompresowaæ. Jeœli w takiej sytuacji odpowiedni 
%kodek nie zostaje odnaleziony, funkcja zostaje przerwana z b³êdem.
%
%Jeœli na dysku, w którym znajduje siê folder zapisu, nie ma 
%odpowiedniej iloœci wolnego miejsca, jeœli jest on chroniony przed 
%zapisem lub jeœli pliku nie mo¿na zapisaæ z innego powodu, funkcja 
%zostaje przerwana z b³êdem.
%
%parametry:
%
%p_sInputAviFile - nazwa pliku wideo (wraz ze œcie¿k¹), z którego ma 
%zostaæ wyciêty fragment filmu. Plik o danej nazwie musi istnieæ. 
%Parametr ten musi byæ typu string. W przeciwnym wypadku funkcja 
%zostaje przerwana z b³êdem.
%
%p_sOutputAviFile - nazwa pliku wideo (wraz ze œcie¿k¹), do którego ma
%zostaæ zapisany wyciêty fragment filmu. Parametr ten musi byæ typu 
%string. W przeciwnym wypadku funkcja zostaje przerwana z b³êdem.
%
%p_oAnnotation - obiekt adnotacji, na podstawie którego wycinany jest 
%fragment filmu. Wyciêty fragment odpowiada przedzia³om czasowym 
%adnotacji okreœlonym polami iStartIndex i iStopIndex. Parametr ten 
%musi byæ obiektem struktury oAnnotation. W przeciwnym wypadku funkcja
%zostaje przerwana z b³êdem.
%
%p_bUseCompression - Parametr ten okreœla, czy zapisywany fragment 
%filmu ma zostaæ skompresowany. Jeœli przyjmuje on wartoœæ true, 
%u¿yta zostaje kompresja Cinepak. Wprzeciwnym wypadku, film nie jest
%kompresowany. Parametr ten musi byæ typu logicznego(wartoœæ true lub
%false). W przeciwnym wypadku funkcja zostaje przerwana z b³êdem.
%
%wartoœæ zwracana:
%
%brak

	if ischar(p_sInputAviFile) == 0 
		disp('Error: p_sInputAviFile is not a string.');
		return;
	end
	
	if ischar(p_sOutputAviFile) == 0 
		disp('Error: p_sOutputAviFile is not a string.');
		return;
	end

	if isstruct(p_oAnnotation) == 0
		disp('Error: p_oAnnotation is not a structure');
		return;
	end
	
	if ~isfield(p_oAnnotation, 'iStartIndex') || ~isfield(p_oAnnotation, 'iStopIndex')
		disp('Error: p_oAnnotation is a wrong structure');
		return;
	end
	
	if islogical(p_bUseCompression) == 0 
		disp('Error: p_bUseCompression is not a logical value.');
		return;
	end
	
	if p_bUseCompression
		sCompression = 'Cinepak';
	else
		sCompression = 'None';
	end
	
	try			
		sInfo = aviinfo(p_sInputAviFile);
		iFPS = sInfo.FramesPerSecond;
		
		iFirstFrame = p_oAnnotation.iStartIndex;
		iLastFrame = p_oAnnotation.iStopIndex;
		
		if iFirstFrame == 0
			iFirstFrame = 1;
		end
		
		if iLastFrame == 0
			iLastFrame = 1;
		end
		
		oInputMovie = mmreader(p_sInputAviFile);
	catch
		sTemp = strcat('Error: Cannot open a file: ''', p_sInputAviFile, '''.');
		disp(sTemp);
		return;
	end
	
	try		
		i=1;
		while iFirstFrame <= iLastFrame
			data = read(oInputMovie,iFirstFrame);
			frames(i).cdata = data;
			frames(i).colormap = [];
			
			i = i+1;
			iFirstFrame = iFirstFrame+1;
		end
	catch
		sTemp = strcat('Error: Cannot read from a file: ''', p_sInputAviFile, '''.');
		disp(sTemp);
		return;
	end
	
	try	
		oOutputMovie = avifile(p_sOutputAviFile,'compression',sCompression,'fps',iFPS);
	catch
		sTemp = strcat('Error: Cannot make a file: ''', p_sOutputAviFile, '''.');
		disp(sTemp);
		return;
	end

	try	
		for i = 1 : numel(frames)
			oOutputMovie = addframe(oOutputMovie,frames(i));
		end
	catch
		sTemp = strcat('Error: Cannot write to file: ''', p_sOutputAviFile, '''.');
		disp(sTemp);
		oOutputMovie = close(oOutputMovie);
		return;
	end
	
	oOutputMovie = close(oOutputMovie);
end






