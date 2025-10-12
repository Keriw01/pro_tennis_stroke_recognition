function oElan = LoadEAF(p_sFile, p_iFrameRate)

%£aduje dane z pliku p_sFile do obiektu struktury oElan i zwraca go. 
%Funkcja ustawia liczbê ramek na sekundê na p_iFrameRate.
%
%parametry:
%
%p_szFile - nazwa pliku (wraz ze œcie¿k¹ dostêpu), z którego maj¹ 
%zostaæ za³adowane dane. Plik o danej nazwie musi istnieæ i posiadaæ 
%rozszerzenie .eaf. Parametr ten musi byæ typu string. W przeciwnym 
%wypadku funkcja zostaje przerwana z b³êdem.
%
%p_iFrameRate - Liczba ramek na sekundê pliku wideo skojarzonego z 
%plikiem p_sFile. Przedzia³y czasowe adnotacji w pliku *.eaf 
%przechowywane s¹ w milisekundach, natomiast funkcja ³aduje je do 
%pamiêci w postaci numerów ramek. Dlatego konieczne jest podanie tego 
%parametru. Parametr ten musi byæ typu numerycznego. W przeciwnym 
%wypadku funkcja zostaje przerwana z b³êdem.
%
%wartoœæ zwracana:
%
%Obiekt struktury oElan z za³adowanymi danymi z pliku *.eaf.

	if ischar(p_sFile) == 0 
		oElan = [];
		disp('Error: p_sFile is not a string.');
		return;
	end
	
	if isnumeric(p_iFrameRate) == 0 
		oElan = [];
		disp('Error: p_iFrameRate is not a number.');
		return;
	end

	if strcmpi(p_sFile(end-3 : end), '.eaf') == 0
		oElan = [];
		sTemp = strcat('Error: Cannot open a file: ''', p_sFile, '''. Wrong extension.');
		disp(sTemp);
		return;
	end
	
	try	
		oFileId = fopen(p_sFile);
		sTextLine = ReadLine(oFileId);
	catch
		oElan = [];
		sTemp = strcat('Error: Cannot open a file: ''', p_sFile, '''.');
		disp(sTemp);
		return;
	end

	%struct elan
	oElan.iFrameRate = p_iFrameRate;
	oElan.iTimeSlots = [];
	oElan.sConstraintDescriptions = {};
	oElan.sConstraintStereotypes = {};
	oElan.sHeaderMediaFile = [];
	oElan.sHeaderTimeUnits = [];
	oElan.sMediaURLs = {};
	oElan.sMimeTypes = {};
	oElan.sRelativeMediaURLs = {};
	oElan.sExtractedFroms = {};
	oElan.sPropertyNames = {};
	oElan.sPropertyValues = {};
	oElan.sDocumentAuthor = [];
	oElan.sDocumentDate = [];
	oElan.sDocumentFormat = [];
	oElan.sDocumentVersion = [];
	oElan.sDocumentXMLNS = 'http://www.w3.org/2001/XMLSchema-instance';
	oElan.sDocumentNoNamespaceSchemaLocation = [];
	oElan.sCountryCodes = {};
	oElan.sLanguageCodes = {};
	oElan.sGraphicReferences = {};
	oElan.sLinguisticTypeIDs = {};
	oElan.sTimeAlignables = {};
	oElan.sLinguisticConstraints = {};
	oElan.sControlledVocabularyRefs = {};
	
		oElan.oControlledVocabularies.sID = [];
		oElan.oControlledVocabularies.sDescription = [];
		oElan.oControlledVocabularies.sEntryDescriptions = {};
		oElan.oControlledVocabularies.sEntryValues = {};
	
		oElan.oTiers.sName = [];
		oElan.oTiers.sDefaultLocale = [];
		oElan.oTiers.sLinguisticType = [];
		oElan.oTiers.sParentRef = [];
		oElan.oTiers.sAnnotator = [];
		oElan.oTiers.sParticipant = [];
	
			oElan.oTiers.oAnnotations.iStartIndex = [];
			oElan.oTiers.oAnnotations.iStopIndex = [];
			oElan.oTiers.oAnnotations.sFields = {};
			oElan.oTiers.oAnnotations.bIsReference = false;
			oElan.oTiers.oAnnotations.sRefTierName = [];
	%struct elan end
	
	while ischar(sTextLine)
		%is time slot entry
		bFound = strfind(sTextLine,'<TIME_SLOT');
		if bFound
			sTimeSlot = GetAttribute(sTextLine,'TIME_VALUE');
			if sTimeSlot
				oElan.iTimeSlots(end+1) = str2num( sTimeSlot );
			else
				oElan.iTimeSlots(end+1) = oElan.iTimeSlots(end);
			end
		end
		%is time slot entry end
		
		%is tier entry
		bFound = strfind(sTextLine,'<TIER');
		if bFound
			oElan.oTiers(end+1).sDefaultLocale = GetAttribute(sTextLine,'DEFAULT_LOCALE');
			oElan.oTiers(end).sLinguisticType = GetAttribute(sTextLine,'LINGUISTIC_TYPE_REF');
			oElan.oTiers(end).sName = GetAttribute(sTextLine,'TIER_ID');
			oElan.oTiers(end).sParentRef = GetAttribute(sTextLine,'PARENT_REF');
			oElan.oTiers(end).sAnnotator = GetAttribute(sTextLine,'ANNOTATOR');
			oElan.oTiers(end).sParticipant = GetAttribute(sTextLine,'PARTICIPANT');
		end
		%is tier entry end
		
		%is annotation entry
		bFound = strfind(sTextLine,'<ALIGNABLE_ANNOTATION');
		if bFound			
			sTemp = GetAttribute(sTextLine,'TIME_SLOT_REF1');
			iTempStart = str2num( sTemp(3:end) );
			
			sTemp = GetAttribute(sTextLine,'TIME_SLOT_REF2');
			iTempStop = str2num( sTemp(3:end) );
			
			iAnnotationStartTime = oElan.iTimeSlots( iTempStart );
			iAnnotationStopTime = oElan.iTimeSlots( iTempStop );
			
			iAnnotationStartFrame = int32(iAnnotationStartTime*p_iFrameRate/1000);
			iAnnotationStopFrame = int32(iAnnotationStopTime*p_iFrameRate/1000);
			
			oElan.oTiers(end).oAnnotations(end+1).bIsReference = false;
			oElan.oTiers(end).oAnnotations(end).sRefTierName = [];
			oElan.oTiers(end).oAnnotations(end).iStartIndex = iAnnotationStartFrame;
			oElan.oTiers(end).oAnnotations(end).iStopIndex = iAnnotationStopFrame;
		end
		%is annotation entry end
		
		%is reference annotation entry
		bFound = strfind(sTextLine,'<REF_ANNOTATION');
		if bFound			
			sTemp = GetAttribute(sTextLine,'ANNOTATION_REF');
			iAnnotationRefNumber = str2num( sTemp(2:end) );
			iTempAnnotationsNumber = 0;
			
			for i=2 : numel(oElan.oTiers)
				if( iAnnotationRefNumber > (numel(oElan.oTiers(i).oAnnotations) + iTempAnnotationsNumber) )
					iTempAnnotationsNumber = iTempAnnotationsNumber + numel(oElan.oTiers(i).oAnnotations);
				else
					oRefAnnotation = oElan.oTiers(i).oAnnotations(iAnnotationRefNumber-iTempAnnotationsNumber);
					sTempTierName = oElan.oTiers(i).sName;
					break;
				end
			end			
			
			oElan.oTiers(end).oAnnotations(end+1).bIsReference = true;
			oElan.oTiers(end).oAnnotations(end).sRefTierName = sTempTierName;
			oElan.oTiers(end).oAnnotations(end).iStartIndex = oRefAnnotation.iStartIndex;
			oElan.oTiers(end).oAnnotations(end).iStopIndex = oRefAnnotation.iStopIndex;			
		end
		%is reference annotation entry end
		
		%is annotation value entry
		bFound = strfind(sTextLine,'<ANNOTATION_VALUE');
		if bFound
			sTemp = GetNodeValue(sTextLine);
			
				if sTemp
					if sTemp(end) == ';'
						sTemp = sTemp(1:end-1);
					end
				end
			
			oPosition = 1;
			while ~isempty(oPosition)
				oPosition = strfind(sTemp,';');
				
				if isempty(oPosition)
					sField = sTemp;
				else
					sField = sTemp(1 : oPosition(1)-1);
					sTemp = sTemp(oPosition(1)+1 : end);
				end	
				
				if isfield(oElan.oTiers(end).oAnnotations(end), 'sFields')
					oElan.oTiers(end).oAnnotations(end).sFields{end+1} = sField;
				else
					oElan.oTiers(end).oAnnotations(end).sFields{1} = sField;
				end
			end
		end
		%is annotation value entry end
		
		%is constraint entry
		bFound = strfind(sTextLine,'<CONSTRAINT');
		if bFound
			oElan.sConstraintDescriptions{end+1} = GetAttribute(sTextLine,'DESCRIPTION');
			oElan.sConstraintStereotypes{end+1} = GetAttribute(sTextLine,'STEREOTYPE');
		end
		%is constraint entry end
		
		%is header entry
		bFound = strfind(sTextLine,'<HEADER');
		if bFound
			oElan.sHeaderMediaFile = GetAttribute(sTextLine,'MEDIA_FILE');
			oElan.sHeaderTimeUnits = GetAttribute(sTextLine,'TIME_UNITS');
		end
		%is header entry end
		
		%is media descriptor entry
		bFound = strfind(sTextLine,'<MEDIA_DESCRIPTOR');
		if bFound
			sTemp = GetAttribute(sTextLine,'MEDIA_URL');
			if sTemp
				if strcmp(sTemp(1:8), 'file:///');
					oElan.sMediaURLs{end+1} = sTemp(9:end);
				else
					oElan.sMediaURLs{end+1} = sTemp;
				end
			else
				oElan.sMediaURLs{end+1} = [];
			end
			
			oElan.sMimeTypes{end+1} = GetAttribute(sTextLine,'MIME_TYPE');
			
			sTemp = GetAttribute(sTextLine,'RELATIVE_MEDIA_URL');
			if sTemp
				if strcmp(sTemp(1:8), 'file:/./');
					oElan.sRelativeMediaURLs{end+1} = sTemp(9:end);
				else
					oElan.sRelativeMediaURLs{end+1} = sTemp;
				end
			else
				oElan.sRelativeMediaURLs{end+1} = [];
			end
			
			oElan.sExtractedFroms{end+1} = GetAttribute(sTextLine,'EXTRACTED_FROM');
		end
		%is media descriptor entry end
		
		%is property entry
		bFound = strfind(sTextLine,'<PROPERTY');
		if bFound
			oElan.sPropertyNames{end+1} = GetAttribute(sTextLine,'NAME');
			
			sPropertyValue = GetNodeValue(sTextLine);
			
			oElan.sPropertyValues{end+1} = sPropertyValue;
		end
		%is property entry end
		
		%is annotation document entry
		bFound = strfind(sTextLine,'<ANNOTATION_DOCUMENT');
		if bFound
			oElan.sDocumentAuthor = GetAttribute(sTextLine,'AUTHOR');
			oElan.sDocumentDate = GetAttribute(sTextLine,'DATE');
			oElan.sDocumentFormat = GetAttribute(sTextLine,'FORMAT');
			oElan.sDocumentVersion = GetAttribute(sTextLine,'VERSION');
			oElan.sDocumentXMLNS = GetAttribute(sTextLine,'xmlns:xsi');
			if isempty(oElan.sDocumentXMLNS)
				oElan.sDocumentXMLNS = 'http://www.w3.org/2001/XMLSchema-instance';
			end
			
			oElan.sDocumentNoNamespaceSchemaLocation = GetAttribute(sTextLine,'xsi:noNamespaceSchemaLocation');
		end
		%is annotation document entry end
				
		%is locale entry
		bFound = strfind(sTextLine,'<LOCALE');
		if bFound
			oElan.sCountryCodes{end+1} = GetAttribute(sTextLine,'COUNTRY_CODE');
			oElan.sLanguageCodes{end+1} = GetAttribute(sTextLine,'LANGUAGE_CODE');
		end
		%is locale entry end
		
		%is linguistic type entry
		bFound = strfind(sTextLine,'<LINGUISTIC_TYPE');
		if bFound
			oElan.sGraphicReferences{end+1} = GetAttribute(sTextLine,'GRAPHIC_REFERENCES');
			oElan.sLinguisticTypeIDs{end+1} = GetAttribute(sTextLine,'LINGUISTIC_TYPE_ID');
			oElan.sTimeAlignables{end+1} = GetAttribute(sTextLine,'TIME_ALIGNABLE');
			oElan.sLinguisticConstraints{end+1} = GetAttribute(sTextLine,'CONSTRAINTS');
			oElan.sControlledVocabularyRefs{end+1} = GetAttribute(sTextLine,'CONTROLLED_VOCABULARY_REF');
		end
		%is linguistic type entry end
		
		%is controlled vocabulary entry
		bFound = strfind(sTextLine,'<CONTROLLED_VOCABULARY');
		if bFound
			oElan.oControlledVocabularies(end+1).sID = GetAttribute(sTextLine,'CV_ID');
			oElan.oControlledVocabularies(end).sDescription = GetAttribute(sTextLine,'DESCRIPTION');
		end
		%is controlled vocabulary entry end
		
		%is cv entry entry entry
		bFound = strfind(sTextLine,'<CV_ENTRY');
		if bFound
			oElan.oControlledVocabularies(end).sEntryDescriptions{end+1} = GetAttribute(sTextLine,'DESCRIPTION');
						
			sEntryValue = GetNodeValue(sTextLine);
			
			oElan.oControlledVocabularies(end).sEntryValues{end+1} = sEntryValue;
		end
		%is cv entry entry entry end
		
		sTextLine = ReadLine(oFileId);
	end
	
	%deleting first empty tier
	oElan.oTiers(1) = [];
	
	%deleting first empty controlled vocabulary
	oElan.oControlledVocabularies(1) = [];

	fclose(oFileId);

end

%private functions
function sAttribute = GetAttribute(lp_sTextLine, lp_sAttributeName)

	sTemp = lp_sTextLine;
	iPositions = strfind(lp_sTextLine, lp_sAttributeName);
	
	if iPositions
		iPosition = iPositions(1);
		sTemp = sTemp(iPosition+length(lp_sAttributeName)+2 : end);
		iEnd = strfind(sTemp, '"');
		sTemp = sTemp(1 : iEnd-1);
		sAttribute = sTemp;
	else
		sAttribute = [];
	end
	
end

function sNodeValue = GetNodeValue(lp_sTextLine)

	iPosition = strfind(lp_sTextLine,'>');
	iPosition2 = strfind(lp_sTextLine,'<');
	
	if numel(iPosition2) >= 2
		sNodeValue = lp_sTextLine(iPosition(1)+1 : iPosition2(2)-1);
	else
		sNodeValue = lp_sTextLine(iPosition(1)+1 : end);
	end
	
end

function sTextLine = ReadLine(oFileId)

	sTextLine = fgets(oFileId);
    if ~ischar(sTextLine)
        return;
    end
	iPosition = strfind(sTextLine,'>');
	while isempty(iPosition)
		sTextLineTemp = fgets(oFileId);
		sTextLine = [sTextLine sTextLineTemp];
		
		iPosition = strfind(sTextLine,'>');
    end
    
end
%private functions end




















