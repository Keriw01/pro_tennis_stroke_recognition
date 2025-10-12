function SaveEAF(p_oElan, p_sFile)

%Zapisuje dane obiektu p_oElan do pliku p_sFile. Jeœli plik o podanej
%nazwie nie istnieje, jest on tworzony od nowa. W przeciwnym wypadku 
%plik jest nadpisywany. Funkcja ta zapisuje w pliku aktualn¹ datê i 
%godzinê (z momentu jej wywo³ania) na podstawie danych zegara 
%systemowego. Jeœli na dysku, w którym znajduje siê folder zapisu, 
%nie ma odpowiedniej iloœci wolnego miejsca, jeœli jest on chroniony 
%przed zapisem lub jeœli pliku nie mo¿na zapisaæ z innego powodu, 
%funkcja zostaje przerwana z b³êdem.
%
%parametry:
%
%p_oElan - obiekt, którego dane maj¹ zostaæ zapisane do pliku *.eaf. 
%Parametr ten musi byæ obiektem struktury oElan. W przeciwnym wypadku 
%funkcja zostaje przerwana z b³êdem.
%
%p_szFile - nazwa pliku (wraz ze œcie¿k¹ dostêpu), do którego maj¹ 
%zostaæ zapisane dane. Jeœli w nazwie nie ma rozszerzenia .eaf, jest 
%ono dodawane. Parametr ten musi byæ typu string. W przeciwnym wypadku
%funkcja zostaje przerwana z b³êdem.
%
%wartoœæ zwracana:
%
%brak

	if isstruct(p_oElan) == 0
		disp('Error: p_oElan is not a structure');
		return;
	end
	
	if ~isfield(p_oElan, 'oTiers') || ~isfield(p_oElan.oTiers, 'oAnnotations') || ~isfield(p_oElan.oTiers, 'sName')
		disp('Error: p_oElan is a wrong structure');
		return;
	end
	
	if ischar(p_sFile) == 0 
		disp('Error: p_sFile is not a string.');
		return;
	end
	
	if strcmpi(p_sFile(end-3 : end), '.eaf') == 0
		p_sFile = strcat(p_sFile, '.eaf');
	end
	
	try	
		oFileId = fopen(p_sFile, 'w');
	catch
		sTemp = strcat('Error: Cannot make a file: ''', p_sFile, '''.');
		disp(sTemp);
		return;
	end
		
	iAnnotationsOverall = 0;
	
	sTemp = '<?xml version="1.0" encoding="ISO-8859-2"?>';
	fprintf(oFileId, '%s\n', sTemp);
	sTemp = strcat('<ANNOTATION_DOCUMENT AUTHOR="', p_oElan.sDocumentAuthor);
	sTemp = strcat(sTemp, '" DATE="', GetLocalTimeAndDate());
	sTemp = strcat(sTemp, '" FORMAT="', p_oElan.sDocumentFormat, '" VERSION="', p_oElan.sDocumentVersion);
	sTemp = strcat(sTemp, '" xmlns:xsi="', p_oElan.sDocumentXMLNS);
	sTemp = strcat(sTemp, '" xsi:noNamespaceSchemaLocation="', p_oElan.sDocumentNoNamespaceSchemaLocation, '">');
	fprintf(oFileId, '%s\n', sTemp);
	sTemp = strcat('<HEADER MEDIA_FILE="', p_oElan.sHeaderMediaFile);
	sTemp = strcat(sTemp, '" TIME_UNITS="', p_oElan.sHeaderTimeUnits, '">');
	fprintf(oFileId, '\t%s\n', sTemp);
	
	for i = 1 : numel(p_oElan.sMediaURLs)
		sTemp = strcat('<MEDIA_DESCRIPTOR MEDIA_URL="file:///', p_oElan.sMediaURLs{i});
		sTemp = strcat(sTemp, '" MIME_TYPE="', p_oElan.sMimeTypes{i});
		
		if numel(p_oElan.sExtractedFroms) >= i && ~isempty(p_oElan.sExtractedFroms{i})
			sTemp = strcat(sTemp, '" EXTRACTED_FROM="', p_oElan.sExtractedFroms{i});
		end
		
		if numel(p_oElan.sRelativeMediaURLs) >= i && ~isempty(p_oElan.sRelativeMediaURLs{i})
			sTemp = strcat(sTemp, '" RELATIVE_MEDIA_URL="file:/./', p_oElan.sRelativeMediaURLs{i});
		end
		
		sTemp = strcat(sTemp, '"/>');
		
		fprintf(oFileId, '\t\t%s\n', sTemp);
	end
	
	for i = 1 : numel(p_oElan.sPropertyNames)
		if ~strcmp(p_oElan.sPropertyNames{i}, 'lastUsedAnnotationId')
			sTemp = strcat('<PROPERTY NAME="', p_oElan.sPropertyValues{i}, '">', p_oElan.sPropertyNames{i}, '</PROPERTY>');
		end
	end
		
	iLastID=0;
	for j = 1 : numel(p_oElan.oTiers)
		iLastID = iLastID + numel(p_oElan.oTiers(j).oAnnotations);
	end
	sTemp = strcat('<PROPERTY NAME="lastUsedAnnotationId">', num2str(iLastID), '</PROPERTY>');
	fprintf(oFileId, '\t\t%s\n', sTemp);
	
	sTemp = '</HEADER>';
	fprintf(oFileId, '\t%s\n', sTemp);
	
	sTemp = '<TIME_ORDER>';
	fprintf(oFileId, '\t%s\n', sTemp);
	
	iTimeSlots = [];
	
	for i = 1 : numel(p_oElan.oTiers)
		for j = 1 : numel(p_oElan.oTiers(i).oAnnotations)
			iTimeSlots(end+1) = int32(p_oElan.oTiers(i).oAnnotations(j).iStartIndex * 1000 / p_oElan.iFrameRate);
			iTimeSlots(end+1) = int32(p_oElan.oTiers(i).oAnnotations(j).iStopIndex * 1000 / p_oElan.iFrameRate);
		end
	end
	
	iTimeSlots = unique(iTimeSlots); %sorts time slots in ascending order and removes repeated elements
	
	for i = 1 : numel(iTimeSlots)
		sTemp = strcat('<TIME_SLOT TIME_SLOT_ID="ts', num2str(i), '" TIME_VALUE="');
		sTemp = strcat(sTemp, num2str(iTimeSlots(i)), '"/>');
		fprintf(oFileId, '\t\t%s\n', sTemp);
	end
	
	sTemp = '</TIME_ORDER>';
	fprintf(oFileId, '\t%s\n', sTemp);
	
	for i = 1 : numel(p_oElan.oTiers)
		oAnnotations = p_oElan.oTiers(i).oAnnotations;
		sTemp = strcat('<TIER DEFAULT_LOCALE="', p_oElan.oTiers(i).sDefaultLocale);
		sTemp = strcat(sTemp, '" LINGUISTIC_TYPE_REF="', p_oElan.oTiers(i).sLinguisticType);
		sTemp = strcat(sTemp, '" TIER_ID="', p_oElan.oTiers(i).sName);
		
		if p_oElan.oTiers(i).sAnnotator
			sTemp = strcat(sTemp, '" ANNOTATOR="', p_oElan.oTiers(i).sAnnotator);
		end
		
		if p_oElan.oTiers(i).sParticipant
			sTemp = strcat(sTemp, '" PARTICIPANT="', p_oElan.oTiers(i).sParticipant);
		end
		
		if p_oElan.oTiers(i).sParentRef
			sTemp = strcat(sTemp, '" PARENT_REF="', p_oElan.oTiers(i).sParentRef);
		end
		
		sTemp = strcat(sTemp,'">');
		
		fprintf(oFileId, '\t%s\n', sTemp);
		
		for j = 1 : numel(oAnnotations)
			bFoundRef = false;
			
			if(oAnnotations(j).bIsReference == true)
				iRefAnnotationNumber = 0;
				for k = 1 : numel(p_oElan.oTiers)
					if bFoundRef == true
						break;
					end
					if strcmp(oAnnotations(j).sRefTierName, p_oElan.oTiers(k).sName) == 0
						iRefAnnotationNumber = iRefAnnotationNumber + numel( p_oElan.oTiers(k).oAnnotations );
					else
						for l = 1 : numel( p_oElan.oTiers(k).oAnnotations )
							if bFoundRef == true
								break;
							end
							if p_oElan.oTiers(k).oAnnotations(l).iStartIndex == oAnnotations(j).iStartIndex
								bFoundRef = true;
								iRefAnnotationNumber = iRefAnnotationNumber + l;
							end
						end %l
					end
				end %k
				
				if bFoundRef
					sTemp = '<ANNOTATION>';
					fprintf(oFileId, '\t\t%s\n', sTemp);
					sTemp = strcat('<REF_ANNOTATION ANNOTATION_ID="a', num2str(iAnnotationsOverall+1));
					sTemp = strcat(sTemp, '" ANNOTATION_REF="a', num2str(iRefAnnotationNumber), '">');
					fprintf(oFileId, '\t\t\t%s\n', sTemp);
					sTemp = '<ANNOTATION_VALUE>';
					fprintf(oFileId, '\t\t\t\t%s', sTemp);
				end
			else		
				iStartIndexNumber = 0;
				iStopIndexNumber = 0;
				for k = 1 : numel(iTimeSlots)
					if iTimeSlots(k) == int32(oAnnotations(j).iStartIndex * 1000 / p_oElan.iFrameRate)
						iStartIndexNumber = k;
					end
					if iTimeSlots(k) == int32(oAnnotations(j).iStopIndex * 1000 / p_oElan.iFrameRate)
						iStopIndexNumber = k;
					end
				end %j
					sTemp = '<ANNOTATION>';
					fprintf(oFileId, '\t\t%s\n', sTemp);
					sTemp = strcat('<ALIGNABLE_ANNOTATION ANNOTATION_ID="a', num2str(iAnnotationsOverall+1));
					sTemp = strcat(sTemp, '" TIME_SLOT_REF1="ts', num2str(iStartIndexNumber));
					sTemp = strcat(sTemp, '" TIME_SLOT_REF2="ts', num2str(iStopIndexNumber), '">');
					fprintf(oFileId, '\t\t\t%s\n', sTemp);
					sTemp = '<ANNOTATION_VALUE>';
					fprintf(oFileId, '\t\t\t\t%s', sTemp);				
			end %end if reference annotation
			
			sTemp = '';
			for k = 1 : numel(oAnnotations(j).sFields)
				sTemp = strcat(sTemp, oAnnotations(j).sFields{k});
				if k ~= numel(oAnnotations(j).sFields)
					sTemp = strcat(sTemp, ';');
				end
			end %k
			fprintf(oFileId, '%s', sTemp);
			
			if oAnnotations(j).bIsReference == true
				if bFoundRef
					sTemp = '</ANNOTATION_VALUE>';
					fprintf(oFileId, '%s\n', sTemp);
					sTemp = '</REF_ANNOTATION>';
					fprintf(oFileId, '\t\t\t%s\n', sTemp);
					sTemp = '</ANNOTATION>';
					fprintf(oFileId, '\t\t%s\n', sTemp);
				end
			else
				sTemp = '</ANNOTATION_VALUE>';
				fprintf(oFileId, '%s\n', sTemp);
				sTemp = '</ALIGNABLE_ANNOTATION>';
				fprintf(oFileId, '\t\t\t%s\n', sTemp);
				sTemp = '</ANNOTATION>';
				fprintf(oFileId, '\t\t%s\n', sTemp);
			end
			
			iAnnotationsOverall = iAnnotationsOverall+1;
		end %j
		sTemp = '</TIER>';
		fprintf(oFileId, '\t%s\n', sTemp);
	end %i
	
	if numel(p_oElan.sLinguisticTypeIDs) == 0
		sTemp = strcat('<LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" ');
		sTemp = strcat(sTemp, 'LINGUISTIC_TYPE_ID="default-lt" ');
		sTemp = strcat(sTemp, 'TIME_ALIGNABLE="true"/>');
		
		fprintf(oFileId, '\t%s\n', sTemp);
	end
	
	for i = 1 : numel(p_oElan.sLinguisticTypeIDs)
		sTemp = strcat('<LINGUISTIC_TYPE GRAPHIC_REFERENCES="', p_oElan.sGraphicReferences{i});
		sTemp = strcat(sTemp, '" LINGUISTIC_TYPE_ID="', p_oElan.sLinguisticTypeIDs{i});
		sTemp = strcat(sTemp, '" TIME_ALIGNABLE="', p_oElan.sTimeAlignables{i});
		
		if numel(p_oElan.sLinguisticConstraints) >= i && ~isempty(p_oElan.sLinguisticConstraints{i})
			sTemp = strcat(sTemp, '" CONSTRAINTS="', p_oElan.sLinguisticConstraints{i});
		end
		
		if numel(p_oElan.sControlledVocabularyRefs) >= i && ~isempty(p_oElan.sControlledVocabularyRefs{i})
			sTemp = strcat(sTemp, '" CONTROLLED_VOCABULARY_REF="', p_oElan.sControlledVocabularyRefs{i});
		end
		
		sTemp = strcat(sTemp, '"/>');
		
		fprintf(oFileId, '\t%s\n', sTemp);
	end
	
	for i = 1 : numel(p_oElan.sCountryCodes)
		sTemp = strcat('<LOCALE COUNTRY_CODE="', p_oElan.sCountryCodes{i});
		sTemp = strcat(sTemp, '" LANGUAGE_CODE="', p_oElan.sLanguageCodes{i}, '"/>');
		
		fprintf(oFileId, '\t%s\n', sTemp);
	end
	
	for i = 1 : numel(p_oElan.sConstraintDescriptions)
		sTemp = strcat('<CONSTRAINT DESCRIPTION="', p_oElan.sConstraintDescriptions{i});
		sTemp = strcat(sTemp, '" STEREOTYPE="', p_oElan.sConstraintStereotypes{i}, '"/>');
		
		fprintf(oFileId, '\t%s\n', sTemp);
	end
	
	for i = 1 : numel(p_oElan.oControlledVocabularies)
		sTemp = strcat('<CONTROLLED_VOCABULARY CV_ID="', p_oElan.oControlledVocabularies(i).sID);
		sTemp = strcat(sTemp, '" DESCRIPTION="', p_oElan.oControlledVocabularies(i).sDescription, '">');
		fprintf(oFileId, '\t%s\n', sTemp);
		
		for j = 1 : numel( p_oElan.oControlledVocabularies(i).sEntryValues )
			sTemp = strcat('<CV_ENTRY DESCRIPTION="', p_oElan.oControlledVocabularies(i).sEntryDescriptions{j});
			sTemp = strcat(sTemp, '">', p_oElan.oControlledVocabularies(i).sEntryValues{j}, '</CV_ENTRY>');
			fprintf(oFileId, '\t\t%s\n', sTemp);
		end
		
		sTemp = '</CONTROLLED_VOCABULARY>';
		fprintf(oFileId, '\t%s\n', sTemp);
	end
	
	sTemp = '</ANNOTATION_DOCUMENT>';
	
	fprintf(oFileId, '%s\n', sTemp);
	
	fclose(oFileId);
	
end %SaveEAF

%private functions
function sLocalTimeAndDate = GetLocalTimeAndDate()

	sTemp = datestr(clock);
	sLocalTimeAndDate = strcat(sTemp(8:11),'-');
	sMonthNumber = GetMonthNumber(sTemp(4:6));
	sLocalTimeAndDate = strcat(sLocalTimeAndDate, sMonthNumber, '-', sTemp(1:2), 'T', sTemp(13:end), '+01:00');
	
end

function sMonthNumber = GetMonthNumber(sMonthName)

	if sMonthName == 'Jan'
		sMonthNumber = '01';
	elseif sMonthName == 'Feb'
		sMonthNumber = '02';
	elseif sMonthName == 'Mar'
		sMonthNumber = '03';
	elseif sMonthName == 'Apr'
		sMonthNumber = '04';
	elseif sMonthName == 'May'
		sMonthNumber = '05';
	elseif sMonthName == 'Jun'
		sMonthNumber = '06';
	elseif sMonthName == 'Jul'
		sMonthNumber = '07';
	elseif sMonthName == 'Aug'
		sMonthNumber = '08';
	elseif sMonthName == 'Sep'
		sMonthNumber = '09';
	elseif sMonthName == 'Oct'
		sMonthNumber = '10';
	elseif sMonthName == 'Nov'
		sMonthNumber = '11';
	else
		sMonthNumber = '12';
	end
	
end
%private functions end






